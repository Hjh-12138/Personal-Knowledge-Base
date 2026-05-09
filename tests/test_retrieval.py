import pytest
from unittest.mock import patch, MagicMock

from knowledge_base.config import Config
from knowledge_base.retrieval import search
from langchain_core.documents import Document


@pytest.fixture
def config():
    return Config(debug=False, top_k=4)


@patch("knowledge_base.retrieval.Chroma")
def test_search_returns_results(mock_chroma, config):
    mock_instance = MagicMock()
    mock_instance.similarity_search.return_value = [
        Document(page_content="Rayleigh scattering explains why the sky is blue.",
                 metadata={"source": "physics.pdf"})
    ]
    mock_chroma.return_value = mock_instance

    results = search("Why is the sky blue?", config)
    assert len(results) == 1
    assert "Rayleigh" in results[0].page_content


@patch("knowledge_base.retrieval.Chroma")
def test_search_empty_results(mock_chroma, config):
    mock_instance = MagicMock()
    mock_instance.similarity_search.return_value = []
    mock_chroma.return_value = mock_instance

    results = search("quantum chromodynamics not in any document", config)
    assert results == []


@patch("knowledge_base.retrieval.Chroma")
def test_search_respects_top_k(mock_chroma, config):
    mock_instance = MagicMock()
    mock_instance.similarity_search.return_value = [
        Document(page_content=f"Doc {i}", metadata={"source": f"doc{i}.pdf"})
        for i in range(10)
    ]
    mock_chroma.return_value = mock_instance

    results = search("test query", config, k=4)
    assert len(results) == 4


@patch("knowledge_base.retrieval.Chroma")
def test_search_error_returns_empty(mock_chroma, config):
    mock_instance = MagicMock()
    mock_instance.similarity_search.side_effect = RuntimeError("Chroma error")
    mock_chroma.return_value = mock_instance

    results = search("test", config)
    assert results == []
