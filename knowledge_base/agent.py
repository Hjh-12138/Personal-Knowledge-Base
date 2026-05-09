import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from .config import Config
from .llm import get_llm
from .retrieval import search

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """You are a research assistant. Break down the following complex research question into 2-4 standalone sub-questions that, when answered individually, will combine to fully address the original question.

Rules:
- Each sub-question should be answerable independently
- Sub-questions should cover different aspects of the original question
- Return ONLY the sub-questions, one per line, numbered 1, 2, 3, 4
- Do NOT answer the questions, just list them

Original question: {question}

Sub-questions:"""

SYNTHESIS_PROMPT = """You are a research assistant synthesizing answers from multiple sub-questions into a comprehensive response.

Original question: {question}

Sub-question results:
{sub_results}

Synthesize a comprehensive answer that:
- Integrates findings from all sub-questions
- Cites specific sources using [1], [2] notation where sources are available
- Notes any contradictions or gaps between sub-answers
- Is structured with clear sections
- Ends with a brief summary

Answer:"""


def decompose_query(question: str, config: Config) -> List[str]:
    """Break a complex research question into 2-4 standalone sub-questions."""
    prompt = DECOMPOSE_PROMPT.format(question=question)

    try:
        llm = get_llm(config)
        response = llm.invoke(prompt).strip()

        sub_questions = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                cleaned = line.lstrip("0123456789.-) ").strip()
                if cleaned and len(cleaned) > 10:
                    sub_questions.append(cleaned)

        if config.debug:
            logger.info("Decomposed '%s' into %d sub-questions", question[:80], len(sub_questions))

        return sub_questions[:4] if sub_questions else [question]
    except Exception as e:
        logger.error("Query decomposition failed: %s", e)
        return [question]


def web_search_tool(query: str, config: Config, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo and return formatted results."""
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}"
                )

        if config.debug:
            logger.info("Web search '%s' -> %d results", query[:80], len(results))

        return "\n\n".join(results) if results else "No web results found."
    except Exception as e:
        logger.error("Web search failed for '%s': %s", query[:80], e)
        return f"Web search unavailable: {e}"


def research_sub_question(
    sub_question: str, config: Config, use_web: bool = False
) -> Dict[str, Any]:
    """Research a single sub-question using local KB and optionally web search."""
    result: Dict[str, Any] = {
        "question": sub_question,
        "kb_results": [],
        "web_results": "",
        "answer": "",
    }

    kb_docs = search(sub_question, config)
    result["kb_results"] = kb_docs

    if use_web:
        result["web_results"] = web_search_tool(sub_question, config)

    from .generation import generate

    if kb_docs:
        result["answer"] = generate(sub_question, kb_docs, config)
    elif result["web_results"] and "No web results" not in result["web_results"]:
        result["answer"] = (
            f"Found web results but no local documents for this question.\n\n"
            f"{result['web_results']}"
        )
    else:
        result["answer"] = "No relevant documents found in knowledge base or web."

    return result


def synthesize_results(
    question: str, sub_results: List[Dict[str, Any]], config: Config
) -> str:
    """Synthesize sub-question results into a comprehensive answer."""
    if len(sub_results) == 1:
        return sub_results[0]["answer"]

    parts = []
    for i, sr in enumerate(sub_results):
        parts.append(f"### Sub-question {i+1}: {sr['question']}")
        parts.append(f"Answer: {sr['answer']}")
        if sr.get("web_results") and "No web results" not in sr["web_results"]:
            parts.append(f"Web results: {sr['web_results'][:500]}")
        parts.append("")

    sub_results_text = "\n".join(parts)
    prompt = SYNTHESIS_PROMPT.format(question=question, sub_results=sub_results_text)

    try:
        llm = get_llm(config)
        response = llm.invoke(prompt).strip()
        return response
    except Exception as e:
        logger.error("Synthesis failed: %s", e)
        fallback = [f"## Research Results for: {question}\n"]
        for i, sr in enumerate(sub_results):
            fallback.append(f"### {sr['question']}")
            fallback.append(sr["answer"])
            fallback.append("")
        return "\n".join(fallback)


def research(
    question: str,
    config: Config,
    tracker: Any = None,
    session_id: Optional[str] = None,
    topic: Optional[str] = None,
    use_web: bool = False,
) -> Dict[str, Any]:
    """Main research entry point.

    Decomposes a complex question into sub-questions, researches each one,
    synthesizes the results, and optionally logs to the learning tracker.

    Returns a dict with:
        - question: original question
        - sub_questions: list of decomposed sub-questions
        - sub_results: list of {question, kb_results, web_results, answer}
        - answer: synthesized final answer
    """
    sub_questions = decompose_query(question, config)

    if config.debug:
        logger.info(
            "Researching %d sub-questions for '%s'", len(sub_questions), question[:80]
        )

    sub_results = []
    for i, sq in enumerate(sub_questions):
        if config.debug:
            logger.info("Sub-question %d/%d: '%s'", i + 1, len(sub_questions), sq[:80])
        sr = research_sub_question(sq, config, use_web=use_web)
        sub_results.append(sr)

    answer = synthesize_results(question, sub_results, config)

    if tracker:
        sources = []
        for sr in sub_results:
            for doc in sr.get("kb_results", []):
                src = doc.metadata.get("source", "")
                if src and src not in sources:
                    sources.append(src)

        try:
            tracker.log_qa(
                question=question,
                answer=answer,
                topic=topic or "General",
                source=", ".join(sources) if sources else None,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Tracker log_qa failed during research: %s", e)

    return {
        "question": question,
        "sub_questions": sub_questions,
        "sub_results": sub_results,
        "answer": answer,
    }
