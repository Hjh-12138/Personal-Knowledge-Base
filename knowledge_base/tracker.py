import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .config import Config

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    first_studied TEXT NOT NULL DEFAULT (datetime('now')),
    last_reviewed TEXT,
    mastery INTEGER NOT NULL DEFAULT 1 CHECK(mastery >= 1 AND mastery <= 5)
);

CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_text TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    topic_name TEXT,
    source_path TEXT,
    session_id TEXT,
    comprehension_notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (topic_name) REFERENCES topics(name)
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    title TEXT,
    topics_covered TEXT,
    read_status TEXT NOT NULL DEFAULT 'unread' CHECK(read_status IN ('unread', 'reading', 'completed')),
    added_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    topic TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    question_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_questions_topic ON questions(topic_name);
CREATE INDEX IF NOT EXISTS idx_questions_session ON questions(session_id);
CREATE INDEX IF NOT EXISTS idx_questions_created ON questions(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_topic ON sessions(topic);
"""


class Tracker:
    def __init__(self, config: Config, db_path: str = None):
        self.config = config
        self.db_path = db_path or str(Path(config.chroma_persist_dir).parent / "learning_tracker.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self):
        self.conn.executescript(SCHEMA_SQL)
        cur = self.conn.execute("SELECT version FROM schema_version")
        row = cur.fetchone()
        if row is None:
            self.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            self.conn.commit()
        elif row[0] < SCHEMA_VERSION:
            for v in range(row[0] + 1, SCHEMA_VERSION + 1):
                logger.info("Running schema migration %d", v)
            self.conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
            self.conn.commit()

    def log_qa(self, question: str, answer: str, topic: str = None, source: str = None,
               session_id: str = None, comprehension_notes: str = None):
        self._log_qa_impl(question, answer, topic, source, session_id, comprehension_notes)

    def _log_qa_impl(self, question, answer, topic, source, session_id, comprehension_notes,
                     _retry=True):
        try:
            # SAVEPOINT 确保全部写入原子成功或全部回滚
            self.conn.execute("SAVEPOINT log_qa_sp")

            if topic:
                self.conn.execute(
                    "INSERT OR IGNORE INTO topics (name, first_studied) VALUES (?, datetime('now'))",
                    (topic,)
                )
                self.conn.execute(
                    "UPDATE topics SET last_reviewed = datetime('now') WHERE name = ?",
                    (topic,)
                )

            if source:
                self.conn.execute(
                    "INSERT OR IGNORE INTO sources (path, title) VALUES (?, ?)",
                    (source, Path(source).stem)
                )

            self.conn.execute(
                """INSERT INTO questions (question_text, answer_text, topic_name, source_path, session_id, comprehension_notes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (question, answer, topic, source, session_id, comprehension_notes)
            )

            if session_id:
                self.conn.execute(
                    "UPDATE sessions SET question_count = question_count + 1 WHERE session_id = ?",
                    (session_id,)
                )

            self._update_mastery(topic)
            self.conn.execute("RELEASE log_qa_sp")
            self.conn.commit()

            if self.config.debug:
                logger.info("Logged Q&A: topic='%s', session='%s'", topic, session_id)
        except sqlite3.Error as e:
            logger.error("Could not save Q&A to learning tracker: %s.", e)
            try:
                self.conn.execute("ROLLBACK TO SAVEPOINT log_qa_sp")
            except sqlite3.Error:
                pass
            if _retry:
                logger.info("Retrying log_qa...")
                time.sleep(0.1)
                self._log_qa_impl(question, answer, topic, source, session_id,
                                  comprehension_notes, _retry=False)
            else:
                logger.error("Retry also failed. Learning record lost for this Q&A.")

    def _update_mastery(self, topic: str):
        if not topic:
            return
        count = self.conn.execute(
            "SELECT COUNT(*) FROM questions WHERE topic_name = ?", (topic,)
        ).fetchone()[0]
        if count <= 1:
            mastery = 1
        elif count <= 3:
            mastery = 2
        elif count <= 6:
            mastery = 3
        elif count <= 10:
            mastery = 4
        else:
            mastery = 5
        self.conn.execute("UPDATE topics SET mastery = ? WHERE name = ?", (mastery, topic))

    def start_session(self, topic: str = None) -> str:
        session_id = f"session_{int(time.time() * 1000)}"
        self.conn.execute(
            "INSERT INTO sessions (session_id, topic) VALUES (?, ?)",
            (session_id, topic)
        )
        self.conn.commit()
        return session_id

    def end_session(self, session_id: str):
        self.conn.execute(
            "UPDATE sessions SET ended_at = datetime('now') WHERE session_id = ?",
            (session_id,)
        )
        self.conn.commit()

    def get_session_history(self, session_id: str) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM questions WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_topics(self) -> List[Dict]:
        rows = self.conn.execute(
            """SELECT t.name, t.mastery, t.last_reviewed, COUNT(q.id) as question_count
               FROM topics t LEFT JOIN questions q ON t.name = q.topic_name
               GROUP BY t.name ORDER BY t.last_reviewed DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        session_count = self.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        question_count = self.conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        topic_count = self.conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        source_count = self.conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        topics = self.list_topics()
        return {
            "session_count": session_count,
            "question_count": question_count,
            "topic_count": topic_count,
            "source_count": source_count,
            "topics": topics,
        }

    def get_knowledge_gaps(self) -> List[Dict]:
        rows = self.conn.execute(
            """SELECT s.path, s.topics_covered
               FROM sources s
               WHERE s.topics_covered IS NOT NULL
               AND s.topics_covered NOT IN (
                   SELECT DISTINCT q.topic_name FROM questions q WHERE q.topic_name IS NOT NULL
               )"""
        ).fetchall()
        return [dict(r) for r in rows]

    def export_topic_markdown(self, topic: str) -> str:
        rows = self.conn.execute(
            "SELECT question_text, answer_text, source_path, created_at FROM questions WHERE topic_name = ? ORDER BY created_at",
            (topic,)
        ).fetchall()
        lines = [
            f"# Topic: {topic}",
            f"Exported: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"Questions: {len(rows)}",
            "",
        ]
        for i, r in enumerate(rows):
            lines.append(f"## Q{i+1}: {r['question_text']}")
            lines.append(f"")
            lines.append(r['answer_text'])
            if r['source_path']:
                lines.append(f"\n*Source: {r['source_path']}*")
            lines.append(f"\n*Asked: {r['created_at']}*")
            lines.append("")
        return "\n".join(lines)

    def close(self):
        self.conn.close()


class LearningTrackerCallback(BaseCallbackHandler):
    def __init__(self, tracker: Tracker, session_id: str = None, topic: str = None):
        self.tracker = tracker
        self.session_id = session_id
        self.topic = topic
        self._last_question: Optional[str] = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        for prompt in prompts:
            if "Question:" in prompt:
                parts = prompt.split("Question:")
                if len(parts) > 1:
                    self._last_question = parts[-1].strip().split("\n")[0].strip()

    def on_llm_end(self, response: LLMResult, **kwargs):
        if not self._last_question:
            return
        answer = ""
        if response.generations and response.generations[0]:
            answer = response.generations[0][0].text

        try:
            self.tracker.log_qa(
                question=self._last_question,
                answer=answer,
                topic=self.topic,
                session_id=self.session_id,
            )
        except Exception as e:
            logger.error("Callback write failed for Q&A: %s", e)
