import streamlit as st
import time
from pathlib import Path

from knowledge_base.config import Config
from knowledge_base.ingestion import load_directory, ingest_directory
from knowledge_base.retrieval import search
from knowledge_base.generation import generate
from knowledge_base.tracker import Tracker
from knowledge_base.agent import research, research_sub_question

st.set_page_config(
    page_title="AI Knowledge Base",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Cached singletons (survive Streamlit reruns)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_config(**overrides):
    cfg = Config.from_env()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


@st.cache_resource
def get_tracker(_config: Config):
    return Tracker(_config)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def init_session():
    defaults = {
        "messages": [],
        "session_id": None,
        "current_topic": None,
        "agent_running": False,
        "agent_sub_results": [],
        "agent_sub_total": 0,
        "agent_sub_done": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()
config = get_config()
tracker = get_tracker(config)

# Start a learning session if none active
if st.session_state.session_id is None:
    st.session_state.session_id = tracker.start_session(
        topic=st.session_state.current_topic or "General"
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ingest_uploaded_file(uploaded) -> int:
    """Save an uploaded file into ./documents/ and re-index. Returns chunk count."""
    docs_dir = Path(config.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / uploaded.name
    dest.write_bytes(uploaded.getbuffer())
    chunk_count, _ = ingest_directory(str(docs_dir), config)
    return chunk_count


def reload_documents():
    """Re-scan the documents directory and return document list."""
    docs_dir = Path(config.documents_dir)
    if not docs_dir.exists():
        return []
    docs, _ = load_directory(str(docs_dir), config)
    return docs


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------

col_top_left, col_top_right = st.columns([3, 1])
with col_top_left:
    st.title("AI Knowledge Base")
with col_top_right:
    backend_label = {
        "ollama": "Ollama",
        "vllm": "vLLM",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
    }.get(config.model_backend, config.model_backend)
    backend_url = config.api_base_url or (
        "http://localhost:8000" if config.model_backend == "vllm" else ""
    )
    st.caption(f"{backend_label}: `{config.model_name}`  {backend_url}")

    if st.button("Export Session"):
        if st.session_state.session_id:
            md = tracker.export_topic_markdown(
                st.session_state.current_topic or "General"
            )
            export_dir = Path("./exports")
            export_dir.mkdir(exist_ok=True)
            export_path = export_dir / f"session_{st.session_state.session_id}.md"
            export_path.write_text(md, encoding="utf-8")
            st.success(f"Exported to {export_path}")

st.divider()

# ---------------------------------------------------------------------------
# Main layout: chat (60%) + dashboard (40%)
# ---------------------------------------------------------------------------

left, right = st.columns([3, 2])

# =========================== LEFT: Chat Panel ==============================

with left:
    st.subheader("Chat")

    # -- Document upload / onboarding --
    docs = reload_documents()
    if not docs:
        st.info(
            "**Welcome to your research assistant!** "
            "Upload documents below or place files in `./documents/` to get started."
        )
        uploaded_files = st.file_uploader(
            "Add documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="doc_uploader",
        )
        if uploaded_files:
            for uf in uploaded_files:
                with st.spinner(f"Ingesting {uf.name}..."):
                    n = ingest_uploaded_file(uf)
                st.success(f"{uf.name}: {n} chunks embedded.")
            st.rerun()
    else:
        with st.expander(f"{len(docs)} documents indexed — upload more"):
            uploaded_files = st.file_uploader(
                "Add documents",
                type=["pdf", "txt", "md"],
                accept_multiple_files=True,
                key="doc_uploader_more",
            )
            if uploaded_files:
                for uf in uploaded_files:
                    with st.spinner(f"Ingesting {uf.name}..."):
                        n = ingest_uploaded_file(uf)
                    st.success(f"{uf.name}: {n} chunks embedded.")
                st.rerun()

    # -- Chat history --
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources"):
                        for s in msg["sources"]:
                            st.caption(f"{s['source']}: _{s['preview']}_")
                if msg.get("sub_questions"):
                    with st.expander("Research steps"):
                        for i, sq in enumerate(msg["sub_questions"]):
                            icon = "" if i < msg.get("sub_done", 0) else "○"
                            st.caption(f"{icon} {sq}")

    # -- Agent progress (visible while researching) --
    if st.session_state.agent_running:
        agent_placeholder = st.empty()

    # -- Chat input --
    use_agent = st.checkbox("Agent mode (decompose + research)", value=False)
    use_web = st.checkbox("Include web search", value=False)

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer_text = ""
            sources = []
            sub_qs = []
            sub_done = 0

            if use_agent:
                # Agentic research with progress
                status_widget = st.status(
                    "Researching...", expanded=True
                )

                from knowledge_base.agent import decompose_query

                with status_widget:
                    st.write("**Decomposing question...**")
                    sub_qs = decompose_query(prompt, config)
                    st.write(f"Found {len(sub_qs)} sub-questions:")
                    for sq in sub_qs:
                        st.caption(f"  {sq}")

                    st.write("**Researching sub-questions...**")
                    sub_results = []
                    progress_bar = st.progress(0)
                    for i, sq in enumerate(sub_qs):
                        st.write(f"_{sq}_")
                        sr = research_sub_question(sq, config, use_web=use_web)
                        sub_results.append(sr)
                        progress_bar.progress((i + 1) / len(sub_qs))
                        if sr["answer"]:
                            st.caption(f"  Found answer ({len(sr['answer'])} chars)")

                    st.write("**Synthesizing...**")
                    from knowledge_base.agent import synthesize_results

                    answer_text = synthesize_results(prompt, sub_results, config)
                    sub_done = len(sub_qs)
                    st.write("Done!")

                # Collect sources
                sources = []
                for sr in sub_results:
                    for doc in sr.get("kb_results", []):
                        src = doc.metadata.get("source", "")
                        preview = doc.page_content[:100].replace("\n", " ")
                        if src and not any(s["source"] == src for s in sources):
                            sources.append({"source": src, "preview": preview})

                st.markdown(answer_text)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "sub_questions": sub_qs,
                        "sub_done": sub_done,
                    }
                )
            else:
                # Single-shot RAG
                with st.spinner("Searching knowledge base..."):
                    docs_found = search(prompt, config)
                    answer_text = generate(prompt, docs_found, config)

                st.markdown(answer_text)

                sources = []
                for doc in docs_found:
                    src = doc.metadata.get("source", "unknown")
                    preview = doc.page_content[:100].replace("\n", " ")
                    sources.append({"source": src, "preview": preview})

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                    }
                )

            # Log to tracker
            topic = st.session_state.current_topic or "General"
            source_paths = [s["source"] for s in sources]
            tracker.log_qa(
                question=prompt,
                answer=answer_text,
                topic=topic,
                source=", ".join(source_paths) if source_paths else None,
                session_id=st.session_state.session_id,
            )

        st.rerun()


# =========================== RIGHT: Dashboard =============================

with right:
    st.subheader("Learning Dashboard")

    stats = tracker.get_stats()

    # -- Topic selector --
    topics = stats.get("topics", [])
    topic_names = [t["name"] for t in topics]

    if not topic_names:
        st.info(
            "Your knowledge base is growing. "
            "Ask a question in the chat to start building your learning profile."
        )
    else:
        selected_topic = st.selectbox(
            "Current topic",
            topic_names,
            index=(
                topic_names.index(st.session_state.current_topic)
                if st.session_state.current_topic in topic_names
                else 0
            ),
        )
        if selected_topic != st.session_state.current_topic:
            st.session_state.current_topic = selected_topic
            st.session_state.session_id = tracker.start_session(topic=selected_topic)

        tab1, tab2, tab3 = st.tabs(["Topics", "Activity", "Knowledge Gaps"])

        # --- Topics tab ---
        with tab1:
            st.caption(f"{stats['topic_count']} topics, {stats['source_count']} sources")

            for t in topics[:10]:
                pct = t["mastery"] / 5
                color = (
                    f"rgb({int(255*(1-pct))}, {int(100+155*pct)}, {int(50*(1-pct))})"
                )
                st.markdown(
                    f"""
<div style="display:flex;align-items:center;gap:8px;margin:4px 0">
  <span style="min-width:120px;font-size:14px">{t['name']}</span>
  <div style="flex:1;height:10px;background:#e0e0e0;border-radius:5px">
    <div style="width:{pct*100}%;height:10px;background:{color};border-radius:5px"></div>
  </div>
  <span style="font-size:12px;color:#888">{t['question_count']} Q</span>
</div>""",
                    unsafe_allow_html=True,
                )

            st.metric("Total Questions", stats["question_count"])
            st.metric("Sessions", stats["session_count"])

        # --- Activity tab ---
        with tab2:
            if st.session_state.session_id:
                history = tracker.get_session_history(st.session_state.session_id)
                st.caption(f"Current session: {len(history)} Q&A pairs")

                for h in reversed(history[-10:]):
                    with st.expander(
                        f"{h['question_text'][:60]}... — {h['created_at'][:16]}"
                    ):
                        st.markdown(f"**Q:** {h['question_text']}")
                        st.markdown(h["answer_text"][:300])
                        if h.get("source_path"):
                            st.caption(f"Source: {h['source_path']}")
            else:
                st.caption("No active session.")

        # --- Knowledge Gaps tab ---
        with tab3:
            gaps = tracker.get_knowledge_gaps()
            if gaps:
                for g in gaps[:5]:
                    topics_covered = g.get("topics_covered", "")
                    st.warning(
                        f"**{g['path']}** covers topics not yet explored: "
                        f"`{topics_covered}`"
                    )
            else:
                st.success("No knowledge gaps detected — keep researching!")

            st.caption(
                "Gaps appear when documents mention topics "
                "you haven't asked questions about yet."
            )
