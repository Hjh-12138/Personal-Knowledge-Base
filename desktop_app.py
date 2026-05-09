#!/usr/bin/env python
"""AI 个人知识库 — 桌面客户端

深色科技风桌面应用，支持文档问答与学习追踪。

依赖: customtkinter, langchain, chromadb, sentence-transformers
启动: python desktop_app.py
"""
import threading
from pathlib import Path

import customtkinter as ctk

from knowledge_base.config import Config
from knowledge_base.ingestion import ingest_directory
from knowledge_base.retrieval import search
from knowledge_base.generation import generate
from knowledge_base.tracker import Tracker
from knowledge_base.agent import (
    decompose_query,
    research_sub_question,
    synthesize_results,
)

# ---------------------------------------------------------------------------
# 主题 — 深色科技风
# ---------------------------------------------------------------------------

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

PALETTE = {
    "bg": "#f0f4ff",
    "card": "#ffffff",
    "card_border": "#d0d8e8",
    "accent": "#2563eb",
    "accent_hover": "#1d4ed8",
    "green": "#16a34a",
    "amber": "#d97706",
    "red": "#dc2626",
    "text": "#1e293b",
    "text_dim": "#64748b",
    "input_bg": "#f8fafc",
}

FONT = ("Microsoft YaHei", 13)
FONT_SMALL = ("Microsoft YaHei", 11)
FONT_MONO = ("Cascadia Code", 13)
FONT_HEADING = ("Microsoft YaHei", 14, "bold")
FONT_TITLE = ("Microsoft YaHei", 19, "bold")

# ---------------------------------------------------------------------------
# 后端单例
# ---------------------------------------------------------------------------

config = Config.from_env()
config_lock = threading.Lock()
tracker = Tracker(config)
session_id = tracker.start_session(topic="通用")
current_topic = "通用"

# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI 个人知识库")
        self.geometry("1200x780")
        self.minsize(900, 600)
        self.configure(fg_color=PALETTE["bg"])

        self._build_topbar()
        self._build_main()
        self._build_bottombar()

        self._load_dashboard()
        self.after(100, self._check_onboarding)

    # ---- 顶栏 --------------------------------------------------------------

    def _build_topbar(self):
        bar = ctk.CTkFrame(self, fg_color=PALETTE["card"], height=44, corner_radius=0)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        ctk.CTkLabel(
            bar, text=" AI 个人知识库",
            font=FONT_TITLE, text_color=PALETTE["accent"],
        ).pack(side="left", padx=(16, 0), pady=8)

        backend_names = {"ollama": "Ollama", "vllm": "vLLM",
                         "openai": "OpenAI", "anthropic": "Anthropic"}
        backend_label = backend_names.get(config.model_backend, config.model_backend)
        self.top_status = ctk.CTkLabel(
            bar,
            text=f" {backend_label}: {config.model_name} ",
            font=FONT_SMALL, text_color=PALETTE["green"],
        )
        self.top_status.pack(side="right", padx=16, pady=8)

        ctk.CTkButton(
            bar, text="设置", width=64, height=28,
            fg_color="transparent", border_color=PALETTE["card_border"],
            border_width=1, font=FONT_SMALL, text_color=PALETTE["text_dim"],
            hover_color=PALETTE["card_border"], command=self._open_settings,
        ).pack(side="right", pady=8, padx=4)

        ctk.CTkButton(
            bar, text="导出", width=64, height=28,
            fg_color="transparent", border_color=PALETTE["card_border"],
            border_width=1, font=FONT_SMALL, text_color=PALETTE["text_dim"],
            hover_color=PALETTE["card_border"], command=self._export,
        ).pack(side="right", pady=8, padx=4)

    # ---- 主体区域 ----------------------------------------------------------

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self._build_chat_panel(main)
        self._build_dashboard(main)

    # -- 对话面板 (左) --

    def _build_chat_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=PALETTE["card"],
                             border_color=PALETTE["card_border"], border_width=1)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        hdr = ctk.CTkFrame(frame, fg_color="transparent", height=36)
        hdr.pack(fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(hdr, text="对话", font=FONT_HEADING,
                     text_color=PALETTE["accent"]).pack(side="left")

        self.chat_box = ctk.CTkTextbox(
            frame, font=FONT_MONO, fg_color=PALETTE["input_bg"],
            text_color=PALETTE["text"], wrap="word",
            border_color=PALETTE["card_border"], border_width=1,
        )
        self.chat_box.pack(fill="both", expand=True, padx=12, pady=8)

        # 预定义对话标签样式 — 避免每次 _do_append 重复创建
        self.chat_box.tag_config("你", foreground=PALETTE["accent"])
        self.chat_box.tag_config("你_body", foreground=PALETTE["text"])
        self.chat_box.tag_config("AI", foreground=PALETTE["green"])
        self.chat_box.tag_config("AI_body", foreground=PALETTE["text"])
        self.chat_box.tag_config("智能体", foreground=PALETTE["text_dim"])
        self.chat_box.tag_config("智能体_body", foreground=PALETTE["text"])
        self.chat_box.tag_config("系统", foreground=PALETTE["text_dim"])
        self.chat_box.tag_config("系统_body", foreground=PALETTE["text"])
        self.chat_box.tag_config("source_tag", foreground=PALETTE["text_dim"])

        # 进度条 — 固定高度占位，避免 pack/pack_forget 引发布局抖动
        self.progress_frame = ctk.CTkFrame(frame, fg_color="transparent", height=28)
        self.progress_frame.pack(fill="x", padx=12)
        self.progress_frame.pack_propagate(False)
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame, width=200, height=8,
            fg_color=PALETTE["card_border"], progress_color=PALETTE["accent"],
        )
        self.progress_bar.pack(side="left", padx=8)
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(
            self.progress_frame, text="", font=FONT_SMALL, text_color=PALETTE["text_dim"],
        )
        self.progress_label.pack(side="left", padx=4)

        # 输入区
        inp_frame = ctk.CTkFrame(frame, fg_color="transparent")
        inp_frame.pack(fill="x", padx=12, pady=(4, 8))

        self.entry = ctk.CTkEntry(
            inp_frame, font=FONT, fg_color=PALETTE["input_bg"],
            text_color=PALETTE["text"], placeholder_text="输入问题...",
            placeholder_text_color=PALETTE["text_dim"], height=36,
            border_color=PALETTE["card_border"],
        )
        self.entry.pack(fill="x", side="top", pady=(0, 4))
        self.entry.bind("<Return>", lambda e: self._ask())

        btn_row = ctk.CTkFrame(inp_frame, fg_color="transparent")
        btn_row.pack(fill="x")

        self.agent_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            btn_row, text="智能体模式", variable=self.agent_var,
            font=FONT_SMALL, text_color=PALETTE["text_dim"],
            fg_color=PALETTE["accent"], border_color=PALETTE["card_border"],
            checkmark_color=PALETTE["bg"],
        ).pack(side="left", padx=4)

        self.web_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            btn_row, text="网络搜索", variable=self.web_var,
            font=FONT_SMALL, text_color=PALETTE["text_dim"],
            fg_color=PALETTE["accent"], border_color=PALETTE["card_border"],
            checkmark_color=PALETTE["bg"],
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_row, text="发送", width=64, height=28,
            fg_color=PALETTE["accent"], hover_color=PALETTE["accent_hover"],
            font=FONT_SMALL, text_color=PALETTE["bg"],
            command=self._ask,
        ).pack(side="right")

    # -- 仪表盘 (右) --

    def _build_dashboard(self, parent):
        dash = ctk.CTkFrame(parent, fg_color=PALETTE["card"],
                            border_color=PALETTE["card_border"], border_width=1)
        dash.grid(row=0, column=1, sticky="nsew")

        hdr = ctk.CTkFrame(dash, fg_color="transparent", height=36)
        hdr.pack(fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(hdr, text="仪表盘", font=FONT_HEADING,
                     text_color=PALETTE["accent"]).pack(side="left")

        self.topic_var = ctk.StringVar(value=current_topic)
        self.topic_menu = ctk.CTkOptionMenu(
            dash, variable=self.topic_var, values=["通用"],
            font=FONT_SMALL, fg_color=PALETTE["input_bg"],
            text_color=PALETTE["text"], button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_hover"],
            dropdown_fg_color=PALETTE["card"],
            dropdown_text_color=PALETTE["text"],
            command=self._on_topic_change,
        )
        self.topic_menu.pack(fill="x", padx=12, pady=(4, 6))

        self.tabview = ctk.CTkTabview(
            dash, fg_color="transparent",
            segmented_button_fg_color=PALETTE["card_border"],
            segmented_button_selected_color=PALETTE["accent"],
            segmented_button_unselected_color=PALETTE["card_border"],
            text_color=PALETTE["text"],
        )
        self.tabview.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.tab_topics = self.tabview.add("主题")
        self.tab_activity = self.tabview.add("动态")
        self.tab_gaps = self.tabview.add("知识缺口")

        # 统计栏
        self.stats_frame = ctk.CTkFrame(dash, fg_color=PALETTE["input_bg"], height=32)
        self.stats_frame.pack(fill="x", padx=12, pady=(0, 8))
        self.stats_label = ctk.CTkLabel(
            self.stats_frame, text="", font=FONT_SMALL, text_color=PALETTE["text_dim"],
        )
        self.stats_label.pack(side="left", padx=10, pady=6)

    # ---- 底栏 --------------------------------------------------------------

    def _build_bottombar(self):
        bar = ctk.CTkFrame(self, fg_color=PALETTE["card"], height=36, corner_radius=0)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        docs = self._count_docs()
        self.doc_label = ctk.CTkLabel(
            bar, text=f" 已索引 {docs} 篇文档",
            font=FONT_SMALL, text_color=PALETTE["text_dim"],
        )
        self.doc_label.pack(side="left", padx=16, pady=6)

        ctk.CTkButton(
            bar, text="重新扫描", width=72, height=24,
            fg_color="transparent", border_color=PALETTE["card_border"],
            border_width=1, font=FONT_SMALL, text_color=PALETTE["text_dim"],
            hover_color=PALETTE["card_border"], command=self._rescan_docs,
        ).pack(side="right", padx=8, pady=6)

        ctk.CTkButton(
            bar, text="上传文档", width=72, height=24,
            fg_color="transparent", border_color=PALETTE["card_border"],
            border_width=1, font=FONT_SMALL, text_color=PALETTE["text_dim"],
            hover_color=PALETTE["card_border"], command=self._upload_docs,
        ).pack(side="right", padx=4, pady=6)

    # ---- 交互操作 ----------------------------------------------------------

    def _ask(self):
        question = self.entry.get().strip()
        if not question:
            return
        self.entry.delete(0, "end")

        self._append_chat("你", question, PALETTE["accent"])

        use_agent = self.agent_var.get()
        use_web = self.web_var.get()

        if use_agent:
            threading.Thread(target=self._agent_ask, args=(question, use_web),
                             daemon=True).start()
        else:
            threading.Thread(target=self._simple_ask, args=(question,),
                             daemon=True).start()

    def _simple_ask(self, question):
        self._set_progress("搜索中...", 0.3)
        docs = search(question, config)
        self._set_progress("生成回答...", 0.6)
        answer = generate(question, docs, config)
        self._set_progress("", 0)
        self._append_chat("AI", answer, PALETTE["green"],
                          sources=[(d.metadata.get("source", "?"),
                                    d.page_content[:120]) for d in docs])
        self._log_and_refresh(question, answer, docs)

    def _agent_ask(self, question, use_web):
        self._set_progress("拆解问题...", 0.1)
        sub_qs = decompose_query(question, config)
        self._append_chat("智能体", f"已将问题拆解为 {len(sub_qs)} 个子问题:\n" +
                          "\n".join(f"  {i+1}. {q}" for i, q in enumerate(sub_qs)),
                          PALETTE["text_dim"])

        sub_results = []
        for i, sq in enumerate(sub_qs):
            self._set_progress(f"[{i+1}/{len(sub_qs)}] {sq[:60]}...", (i + 0.5) / len(sub_qs))
            sr = research_sub_question(sq, config, use_web=use_web)
            sub_results.append(sr)

        self._set_progress("综合回答...", 0.9)
        answer = synthesize_results(question, sub_results, config)
        self._clear_progress()

        all_sources = []
        for sr in sub_results:
            for d in sr.get("kb_results", []):
                src = d.metadata.get("source", "?")
                if not any(s[0] == src for s in all_sources):
                    all_sources.append((src, d.page_content[:120]))

        self._append_chat("AI", answer, PALETTE["green"], sources=all_sources)

        # 收集所有子问题的文档用于日志记录
        all_docs = []
        for sr in sub_results:
            all_docs.extend(sr.get("kb_results", []))
        self._log_and_refresh(question, answer, all_docs)

    def _log_and_refresh(self, question, answer, docs):
        sources = list({d.metadata.get("source", "") for d in docs if d.metadata.get("source")})
        tracker.log_qa(
            question=question, answer=answer,
            topic=self.topic_var.get(), source=", ".join(sources) if sources else None,
            session_id=session_id,
        )
        self.after(0, self._load_dashboard)

    # ---- 对话显示 ----------------------------------------------------------

    def _append_chat(self, sender, text, color, sources=None):
        self.after(0, lambda: self._do_append(sender, text, color, sources))

    def _do_append(self, sender, text, color, sources):
        self.chat_box.configure(state="normal")

        tag = sender
        self.chat_box.insert("end", f"\n  [{sender}]\n", tag)
        for line in text.split("\n"):
            self.chat_box.insert("end", f"  {line}\n", f"{tag}_body")

        if sources:
            self.chat_box.insert("end", "  ── 来源引用 ──\n", "source_tag")
            for i, (src, preview) in enumerate(sources):
                p = preview.replace("\n", " ")
                self.chat_box.insert("end", f"  [{i+1}] {src}: {p}...\n", "source_tag")

        self.chat_box.insert("end", "\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    # ---- 进度条 ------------------------------------------------------------

    def _clear_progress(self):
        self.after(0, lambda: (
            self.progress_label.configure(text=""),
            self.progress_bar.set(0),
        ))

    def _set_progress(self, text, value):
        self.after(0, lambda: (
            self.progress_label.configure(text=text),
            self.progress_bar.set(value),
        ))

    # ---- 仪表盘刷新 --------------------------------------------------------

    def _load_dashboard(self):
        self._refresh_topics()
        self._refresh_activity()
        self._refresh_gaps()
        self._refresh_stats()

    def _refresh_topics(self):
        for w in self.tab_topics.winfo_children():
            w.destroy()

        topics = tracker.list_topics()
        names = [t["name"] for t in topics]
        if names:
            self.topic_menu.configure(values=names)

        if not topics:
            ctk.CTkLabel(
                self.tab_topics, text="暂无主题。\n在对话中提问即可开始构建知识图谱。",
                font=FONT_SMALL, text_color=PALETTE["text_dim"],
            ).pack(pady=20)
            return

        for t in topics[:8]:
            row = ctk.CTkFrame(self.tab_topics, fg_color="transparent", height=28)
            row.pack(fill="x", padx=8, pady=2)

            ctk.CTkLabel(row, text=t["name"], font=FONT_SMALL,
                         text_color=PALETTE["text"], width=100, anchor="w",
                         ).pack(side="left")

            bar = ctk.CTkProgressBar(row, width=120, height=8,
                                     fg_color=PALETTE["card_border"])
            bar.pack(side="left", padx=8)
            bar.set(t["mastery"] / 5.0)
            bar.configure(progress_color=self._mastery_color(t["mastery"]))

            ctk.CTkLabel(row, text=f"{t['mastery']}/5",
                         font=FONT_SMALL, text_color=PALETTE["text_dim"], width=30,
                         ).pack(side="left")

            ctk.CTkLabel(row, text=f"{t['question_count']}问",
                         font=FONT_SMALL, text_color=PALETTE["text_dim"], width=30,
                         ).pack(side="right")

    def _refresh_activity(self):
        for w in self.tab_activity.winfo_children():
            w.destroy()

        history = tracker.get_session_history(session_id)
        if not history:
            ctk.CTkLabel(
                self.tab_activity, text="当前会话暂无活动。",
                font=FONT_SMALL, text_color=PALETTE["text_dim"],
            ).pack(pady=20)
            return

        for h in reversed(history[-10:]):
            frame = ctk.CTkFrame(self.tab_activity, fg_color=PALETTE["input_bg"],
                                 border_color=PALETTE["card_border"], border_width=1)
            frame.pack(fill="x", padx=8, pady=2)

            ctk.CTkLabel(
                frame, text=h["question_text"][:70], font=FONT_SMALL,
                text_color=PALETTE["text"], anchor="w",
            ).pack(fill="x", padx=8, pady=(4, 0))

            ctk.CTkLabel(
                frame, text=h["created_at"][:16], font=FONT_SMALL,
                text_color=PALETTE["text_dim"],
            ).pack(fill="x", padx=8, pady=(0, 4))

    def _refresh_gaps(self):
        for w in self.tab_gaps.winfo_children():
            w.destroy()

        gaps = tracker.get_knowledge_gaps()
        if not gaps:
            ctk.CTkLabel(
                self.tab_gaps, text="未发现知识缺口。\n继续深入研究吧！",
                font=FONT_SMALL, text_color=PALETTE["green"],
            ).pack(pady=20)
            return

        for g in gaps[:5]:
            card = ctk.CTkFrame(self.tab_gaps, fg_color=PALETTE["input_bg"],
                                border_color=PALETTE["amber"], border_width=1)
            card.pack(fill="x", padx=8, pady=2)

            ctk.CTkLabel(
                card, text=f" {g['path']}", font=FONT_SMALL,
                text_color=PALETTE["amber"], anchor="w",
            ).pack(fill="x", padx=8, pady=(4, 0))

            topics_text = g.get("topics_covered", "") or "未标记主题"
            ctk.CTkLabel(
                card, text=f"   覆盖 {topics_text} — 尚未探索",
                font=FONT_SMALL, text_color=PALETTE["text_dim"],
            ).pack(fill="x", padx=8, pady=(0, 4))

    def _refresh_stats(self):
        stats = tracker.get_stats()
        self.stats_label.configure(
            text=f" {stats['question_count']} 次提问  |  "
                 f"{stats['session_count']} 个会话  |  "
                 f"{stats['topic_count']} 个主题  |  "
                 f"{stats['source_count']} 篇来源"
        )

    # ---- 工具方法 ----------------------------------------------------------

    def _mastery_color(self, level):
        if level <= 1: return "#ff5252"
        if level <= 2: return "#ffab00"
        if level <= 3: return "#ffd740"
        if level <= 4: return "#69f0ae"
        return "#00e676"

    def _count_docs(self):
        p = Path(config.documents_dir)
        if not p.exists():
            return 0
        return len([f for f in p.iterdir()
                    if f.suffix.lower() in (".pdf", ".txt", ".md")])

    def _on_topic_change(self, val):
        global current_topic, session_id
        current_topic = val
        session_id = tracker.start_session(topic=val)
        self._refresh_activity()

    def _open_settings(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("模型设置")
        dialog.geometry("420x440")
        dialog.resizable(False, False)
        dialog.configure(fg_color=PALETTE["card"])
        dialog.grab_set()
        dialog.transient(self)

        # 标题
        ctk.CTkLabel(
            dialog, text="模型后端配置", font=FONT_HEADING,
            text_color=PALETTE["accent"],
        ).pack(pady=(16, 12))

        # 后端选择
        ctk.CTkLabel(dialog, text="后端类型", font=FONT_SMALL,
                     text_color=PALETTE["text"]).pack(anchor="w", padx=24)
        backend_var = ctk.StringVar(value=config.model_backend)
        backend_menu = ctk.CTkOptionMenu(
            dialog, variable=backend_var,
            values=["ollama", "vllm", "openai", "anthropic"],
            font=FONT_SMALL, fg_color=PALETTE["input_bg"],
            text_color=PALETTE["text"], button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_hover"],
            dropdown_fg_color=PALETTE["card"],
            dropdown_text_color=PALETTE["text"],
        )
        backend_menu.pack(fill="x", padx=24, pady=(2, 10))

        # 模型名称
        ctk.CTkLabel(dialog, text="模型名称", font=FONT_SMALL,
                     text_color=PALETTE["text"]).pack(anchor="w", padx=24)
        model_var = ctk.StringVar(value=config.model_name)
        model_entry = ctk.CTkEntry(
            dialog, textvariable=model_var, font=FONT_SMALL,
            fg_color=PALETTE["input_bg"], text_color=PALETTE["text"],
            border_color=PALETTE["card_border"],
        )
        model_entry.pack(fill="x", padx=24, pady=(2, 10))

        # API Base URL
        ctk.CTkLabel(dialog, text="API 地址 (vLLM/OpenAI 兼容)", font=FONT_SMALL,
                     text_color=PALETTE["text"]).pack(anchor="w", padx=24)
        url_var = ctk.StringVar(value=config.api_base_url)
        url_entry = ctk.CTkEntry(
            dialog, textvariable=url_var, font=FONT_SMALL,
            fg_color=PALETTE["input_bg"], text_color=PALETTE["text"],
            border_color=PALETTE["card_border"],
            placeholder_text="http://localhost:8000/v1",
            placeholder_text_color=PALETTE["text_dim"],
        )
        url_entry.pack(fill="x", padx=24, pady=(2, 10))

        # API Key
        ctk.CTkLabel(dialog, text="API Key (OpenAI/Anthropic)", font=FONT_SMALL,
                     text_color=PALETTE["text"]).pack(anchor="w", padx=24)
        key_var = ctk.StringVar(value=config.api_key)
        key_entry = ctk.CTkEntry(
            dialog, textvariable=key_var, font=FONT_SMALL,
            fg_color=PALETTE["input_bg"], text_color=PALETTE["text"],
            border_color=PALETTE["card_border"], show="*",
            placeholder_text="sk-...",
            placeholder_text_color=PALETTE["text_dim"],
        )
        key_entry.pack(fill="x", padx=24, pady=(2, 10))

        # 温度
        ctk.CTkLabel(dialog, text=f"温度: {config.temperature:.1f}", font=FONT_SMALL,
                     text_color=PALETTE["text"]).pack(anchor="w", padx=24)
        temp_var = ctk.DoubleVar(value=config.temperature)
        temp_slider = ctk.CTkSlider(
            dialog, variable=temp_var, from_=0.0, to=1.0,
            number_of_steps=20, button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_hover"],
            fg_color=PALETTE["card_border"], progress_color=PALETTE["accent"],
        )
        temp_slider.pack(fill="x", padx=24, pady=(2, 2))

        def do_save():
            with config_lock:
                config.model_backend = backend_var.get()
                config.model_name = model_var.get().strip() or config.model_name
                config.api_base_url = url_var.get().strip()
                config.api_key = key_var.get().strip()
                config.temperature = float(temp_var.get())

            backend_names = {"ollama": "Ollama", "vllm": "vLLM",
                             "openai": "OpenAI", "anthropic": "Anthropic"}
            self.top_status.configure(
                text=f" {backend_names.get(config.model_backend, config.model_backend)}: {config.model_name} "
            )
            dialog.destroy()

        ctk.CTkButton(
            dialog, text="保存", width=80, height=32,
            fg_color=PALETTE["accent"], hover_color=PALETTE["accent_hover"],
            font=FONT_SMALL, text_color="#ffffff", command=do_save,
        ).pack(pady=(16, 8))

    def _export(self):
        md = tracker.export_topic_markdown(self.topic_var.get())
        export_dir = Path("./exports")
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / f"session_{session_id}.md"
        export_path.write_text(md, encoding="utf-8")
        self._append_chat("系统", f"已导出到 {export_path}", PALETTE["text_dim"])

    def _upload_docs(self):
        from tkinter import filedialog, messagebox
        files = filedialog.askopenfilenames(
            title="选择文档",
            filetypes=[("文档", "*.pdf *.txt *.md"), ("所有文件", "*.*")],
        )
        if files:
            docs_dir = Path(config.documents_dir)
            docs_dir.mkdir(parents=True, exist_ok=True)
            overwritten = []
            for f in files:
                dest = docs_dir / Path(f).name
                if dest.exists():
                    overwritten.append(dest.name)
                Path(f).replace(dest)
            if overwritten:
                self._append_chat("系统",
                    f"已覆盖 {len(overwritten)} 个同名文件: {', '.join(overwritten[:5])}",
                    PALETTE["text_dim"])
            self._rescan_docs()

    def _rescan_docs(self):
        self._set_progress("正在索引文档...", 0.3)
        threading.Thread(target=self._do_rescan, daemon=True).start()

    def _do_rescan(self):
        docs_dir = Path(config.documents_dir)
        if docs_dir.exists() and any(docs_dir.iterdir()):
            n, skipped = ingest_directory(str(docs_dir), config)
            self._set_progress("", 0)
            self._append_chat("系统", f"已索引 {n} 个片段（跳过 {skipped} 个文件）",
                              PALETTE["text_dim"])
        else:
            self._set_progress("", 0)
            if not docs_dir.exists():
                docs_dir.mkdir(parents=True, exist_ok=True)
            self._append_chat("系统", "文档目录为空，请先上传文档。", PALETTE["text_dim"])
        docs = self._count_docs()
        self.after(0, lambda: self.doc_label.configure(text=f" 已索引 {docs} 篇文档"))

    def _check_onboarding(self):
        if self._count_docs() == 0:
            self._append_chat(
                "系统",
                "欢迎使用 AI 个人知识库！\n\n"
                "快速开始:\n"
                "  1. 点击 [上传文档] 添加 PDF/TXT/MD 文件\n"
                "  2. 或将文件放入 ./documents/ 目录后点击 [重新扫描]\n"
                "  3. 在输入框中输入问题，按回车发送\n\n"
                "开启「智能体模式」可自动拆解复杂问题，逐步深入研究。",
                PALETTE["text_dim"],
            )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
