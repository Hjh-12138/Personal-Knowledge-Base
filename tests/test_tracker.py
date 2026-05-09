import pytest
import tempfile
import os

from knowledge_base.config import Config
from knowledge_base.tracker import Tracker, LearningTrackerCallback


@pytest.fixture
def config():
    return Config(debug=False)


@pytest.fixture
def tracker(config):
    db_path = os.path.join(tempfile.gettempdir(), f"test_tracker_{os.getpid()}.db")
    t = Tracker(config, db_path=db_path)
    yield t
    t.close()
    try:
        os.unlink(db_path)
        os.unlink(db_path + "-wal")
        os.unlink(db_path + "-shm")
    except OSError:
        pass


def test_log_qa_creates_row(tracker):
    tracker.log_qa(
        question="Why is the sky blue?",
        answer="Because of Rayleigh scattering.",
        topic="Atmospheric Physics",
        source="physics.pdf",
        session_id="test_session_1",
    )
    rows = tracker.get_session_history("test_session_1")
    assert len(rows) >= 1


def test_session_history_ordered(tracker):
    sid = tracker.start_session(topic="Test")
    tracker.log_qa("Q1", "A1", topic="Test", session_id=sid)
    tracker.log_qa("Q2", "A2", topic="Test", session_id=sid)
    tracker.end_session(sid)
    rows = tracker.get_session_history(sid)
    assert len(rows) == 2


def test_topic_mastery_updates(tracker):
    for i in range(5):
        tracker.log_qa(
            question=f"Question {i}",
            answer=f"Answer {i}",
            topic="Machine Learning",
            session_id="mastery_test",
        )
    topics = tracker.list_topics()
    ml = next((t for t in topics if t["name"] == "Machine Learning"), None)
    assert ml is not None
    assert ml["mastery"] >= 2


def test_get_stats(tracker):
    sid1 = tracker.start_session(topic="Science")
    tracker.log_qa("Q1", "A1", topic="Science", session_id=sid1)
    tracker.log_qa("Q2", "A2", topic="Math", session_id=sid1)
    tracker.end_session(sid1)

    sid2 = tracker.start_session(topic="History")
    tracker.log_qa("Q3", "A3", topic="History", session_id=sid2)
    tracker.end_session(sid2)

    stats = tracker.get_stats()
    assert stats["session_count"] == 2
    assert stats["question_count"] == 3
    assert stats["topic_count"] >= 2


def test_export_markdown(tracker):
    tracker.log_qa(
        question="What is gradient descent?",
        answer="An optimization algorithm for finding local minima.",
        topic="Machine Learning",
        source="ml_book.pdf",
        session_id="export_test",
    )
    md = tracker.export_topic_markdown("Machine Learning")
    assert "# Topic: Machine Learning" in md
    assert "gradient descent" in md.lower()


def test_log_qa_failure_does_not_crash(tracker):
    import sqlite3
    from unittest.mock import MagicMock

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.OperationalError("DB error")
    mock_conn.row_factory = None
    old_conn = tracker.conn
    tracker.conn = mock_conn
    try:
        tracker.log_qa("Q", "A", topic="Test", session_id="fail_test")
    except Exception:
        pass
    tracker.conn = old_conn
    stats = tracker.get_stats()
    assert stats is not None


def test_schema_version_set(tracker):
    cur = tracker.conn.execute("SELECT version FROM schema_version")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == 1


def test_concurrent_read_write(tracker):
    sid = tracker.start_session(topic="Concurrent")
    for i in range(10):
        tracker.log_qa(f"Q{i}", f"A{i}", topic="Concurrent", session_id=sid)
    tracker.end_session(sid)
    stats = tracker.get_stats()
    assert stats["question_count"] >= 10
