import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from knowledge_base.config import Config
from knowledge_base.ingestion import load_document, chunk_documents, ingest_directory
from knowledge_base.retrieval import search
from knowledge_base.generation import generate
from knowledge_base.tracker import Tracker
from langchain_core.documents import Document


@pytest.fixture
def config():
    return Config(debug=False, chroma_persist_dir=tempfile.mkdtemp())


@pytest.fixture
def temp_docs_dir():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "physics.txt"), "w", encoding="utf-8") as f:
            f.write(
                "Rayleigh scattering is the elastic scattering of light by particles "
                "much smaller than the wavelength of the radiation. It explains why "
                "the sky appears blue during the day. The scattering intensity is "
                "inversely proportional to the fourth power of wavelength. "
                "This means shorter wavelengths (blue) scatter much more than "
                "longer wavelengths (red). " * 20
            )
        with open(os.path.join(d, "ml.txt"), "w", encoding="utf-8") as f:
            f.write(
                "Gradient descent is a first-order iterative optimization algorithm "
                "for finding local minima of differentiable functions. "
                "The idea is to take repeated steps in the opposite direction "
                "of the gradient of the function at the current point. "
                "Learning rate determines the step size. " * 20
            )
        yield d


@patch("knowledge_base.retrieval.Chroma")
@patch("knowledge_base.ingestion.HuggingFaceEmbeddings")
@patch("knowledge_base.generation.get_llm")
def test_full_pipeline(mock_ollama, mock_embeddings, mock_chroma, config, temp_docs_dir):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        "The sky is blue because of Rayleigh scattering, which causes shorter "
        "wavelengths (blue) to scatter more than longer wavelengths [1]."
    )
    mock_ollama.return_value = mock_llm

    mock_emb = MagicMock()
    mock_embeddings.return_value = mock_emb

    mock_store = MagicMock()
    mock_chroma.return_value = mock_store

    docs = []
    for txt_file in os.listdir(temp_docs_dir):
        docs.extend(load_document(os.path.join(temp_docs_dir, txt_file), config))

    assert len(docs) >= 2

    chunks = chunk_documents(docs, config)
    assert len(chunks) >= 2

    mock_store.similarity_search.return_value = [
        Document(
            page_content="Rayleigh scattering is the elastic scattering of light...",
            metadata={"source": os.path.join(temp_docs_dir, "physics.txt")}
        )
    ]

    answer = generate("Why is the sky blue?", [mock_store.similarity_search.return_value[0]], config)
    assert "Rayleigh" in answer
    assert "[1]" in answer


@patch("knowledge_base.retrieval.Chroma")
@patch("knowledge_base.ingestion.HuggingFaceEmbeddings")
@patch("knowledge_base.generation.get_llm")
def test_e2e_with_tracker(mock_ollama, mock_embeddings, mock_chroma, config, temp_docs_dir):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        "The sky is blue because of Rayleigh scattering [1].",
        "Gradient descent uses learning rate to control step size [2].",
        "Rayleigh scattering explains atmospheric optics [1].",
    ]
    mock_ollama.return_value = mock_llm

    mock_emb = MagicMock()
    mock_embeddings.return_value = mock_emb

    mock_store = MagicMock()
    mock_store.similarity_search.side_effect = [
        [Document(page_content="Rayleigh scattering...", metadata={"source": "physics.txt"})],
        [Document(page_content="Gradient descent...", metadata={"source": "ml.txt"})],
        [Document(page_content="Rayleigh scattering...", metadata={"source": "physics.txt"})],
    ]
    mock_chroma.return_value = mock_store

    db_path = os.path.join(tempfile.gettempdir(), f"test_e2e_{os.getpid()}.db")
    tracker = Tracker(config, db_path=db_path)
    sid = tracker.start_session(topic="Science")

    qa_pairs = [
        ("Why is the sky blue?", "Atmospheric Physics"),
        ("What is gradient descent?", "Machine Learning"),
        ("What is Rayleigh scattering?", "Atmospheric Physics"),
    ]

    for question, topic in qa_pairs:
        docs = search(question, config)
        answer = generate(question, docs, config)
        tracker.log_qa(question=question, answer=answer, topic=topic, session_id=sid)

    tracker.end_session(sid)
    stats = tracker.get_stats()
    assert stats["question_count"] == 3

    tracker.close()
    for suffix in ["", "-wal", "-shm"]:
        try:
            os.unlink(db_path + suffix)
        except OSError:
            pass


@patch("knowledge_base.retrieval.Chroma")
@patch("knowledge_base.ingestion.HuggingFaceEmbeddings")
def test_markdown_export_integration(mock_embeddings, mock_chroma, config, temp_docs_dir):
    mock_emb = MagicMock()
    mock_embeddings.return_value = mock_emb

    db_path = os.path.join(tempfile.gettempdir(), f"test_export_{os.getpid()}.db")
    tracker = Tracker(config, db_path=db_path)

    for i in range(5):
        tracker.log_qa(
            question=f"ML question {i}",
            answer=f"ML answer {i} with source [1].",
            topic="Machine Learning",
            session_id="export_session",
        )

    md = tracker.export_topic_markdown("Machine Learning")
    assert "Machine Learning" in md
    for i in range(5):
        assert f"ML question {i}" in md

    tracker.close()
    for suffix in ["", "-wal", "-shm"]:
        try:
            os.unlink(db_path + suffix)
        except OSError:
            pass
