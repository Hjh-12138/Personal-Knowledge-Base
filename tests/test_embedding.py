import pytest
from unittest.mock import patch, MagicMock

from knowledge_base.config import Config
from knowledge_base.embedding import embed_chunks


@pytest.fixture
def config():
    return Config(debug=False)


@patch("knowledge_base.ingestion.HuggingFaceEmbeddings")
def test_embed_chunks_returns_correct_dimensions(mock_embeddings, config):
    mock_instance = MagicMock()
    mock_instance.embed_documents.return_value = [[0.1] * 384 for _ in range(5)]
    mock_embeddings.return_value = mock_instance

    chunks = [f"Chunk {i} with some text content for testing purposes." for i in range(5)]
    vectors = embed_chunks(chunks, config)
    assert len(vectors) == 5
    assert all(len(v) == 384 for v in vectors)
