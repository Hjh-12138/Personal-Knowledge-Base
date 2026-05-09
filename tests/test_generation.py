import pytest
from unittest.mock import patch, MagicMock

from knowledge_base.config import Config
from knowledge_base.generation import generate, _format_context
from langchain_core.documents import Document


@pytest.fixture
def config():
    return Config(debug=False, model_name="test-model", timeout=30)


def test_generate_empty_context_returns_abstention(config):
    result = generate("Any question?", [], config)
    assert "No relevant documents" in result


def test_format_context_includes_sources():
    docs = [
        Document(page_content="The sky is blue because of Rayleigh scattering.",
                 metadata={"source": "atmosphere.pdf"}),
        Document(page_content="Scattering intensity is proportional to 1/λ⁴.",
                 metadata={"source": "optics.pdf"}),
    ]
    ctx = _format_context(docs)
    assert "[1]" in ctx
    assert "[2]" in ctx
    assert "atmosphere.pdf" in ctx
    assert "optics.pdf" in ctx


@patch("knowledge_base.generation.get_llm")
def test_generate_returns_cited_answer(mock_ollama, config):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "The sky is blue because of Rayleigh scattering [1]."
    mock_ollama.return_value = mock_llm

    docs = [Document(page_content="Rayleigh scattering makes the sky blue.",
                     metadata={"source": "physics.pdf"})]
    result = generate("Why is the sky blue?", docs, config)
    assert "Rayleigh" in result


@patch("knowledge_base.generation.get_llm")
def test_generate_timeout_graceful(mock_ollama, config):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("timed out")
    mock_ollama.return_value = mock_llm

    docs = [Document(page_content="test content", metadata={"source": "test.txt"})]
    result = generate("question", docs, config)
    assert "timed out" in result.lower() or "timeout" in result.lower()


@patch("knowledge_base.generation.get_llm")
def test_generate_connection_error(mock_ollama, config):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Connection refused")
    mock_ollama.return_value = mock_llm

    docs = [Document(page_content="test", metadata={"source": "test.txt"})]
    result = generate("question", docs, config)
    assert "ollama" in result.lower()
