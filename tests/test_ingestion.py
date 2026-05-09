import pytest
import tempfile
import os
from pathlib import Path

from knowledge_base.config import Config
from knowledge_base.ingestion import (
    load_document,
    load_directory,
    chunk_documents,
    _validate_document,
    _make_chunk_id,
    ingest_directory,
)
from langchain_core.documents import Document


@pytest.fixture
def config():
    return Config(debug=False)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_validate_document_with_content(config):
    doc = Document(page_content="Hello world", metadata={"source": "test.txt"})
    result = _validate_document(doc, "test.txt")
    assert result is not None
    assert result.page_content == "Hello world"


def test_validate_document_empty_content(config):
    doc = Document(page_content="   ", metadata={"source": "empty.txt"})
    result = _validate_document(doc, "empty.txt")
    assert result is None


def test_validate_document_empty_string(config):
    doc = Document(page_content="", metadata={"source": "empty.txt"})
    result = _validate_document(doc, "empty.txt")
    assert result is None


def test_make_chunk_id_deterministic():
    id1 = _make_chunk_id("docs/paper.pdf", 0)
    id2 = _make_chunk_id("docs/paper.pdf", 0)
    assert id1 == id2
    assert len(id1) == 32


def test_make_chunk_id_unique_per_index():
    id1 = _make_chunk_id("docs/paper.pdf", 0)
    id2 = _make_chunk_id("docs/paper.pdf", 1)
    assert id1 != id2


def test_chunk_documents(config):
    docs = [Document(page_content="This is a test document. " * 200, metadata={"source": "test.txt"})]
    chunks = chunk_documents(docs, config)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.page_content) <= config.chunk_size + config.chunk_overlap


def test_chunk_short_document(config):
    docs = [Document(page_content="Short text.", metadata={"source": "short.txt"})]
    chunks = chunk_documents(docs, config)
    assert len(chunks) == 1


def test_load_txt_file(config, temp_dir):
    file_path = os.path.join(temp_dir, "test.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Hello world! This is a test document with content.")
    docs = load_document(file_path, config)
    assert len(docs) >= 1
    assert len(docs[0].page_content) > 0
    assert docs[0].metadata["source"] == file_path


def test_load_empty_file(config, temp_dir):
    file_path = os.path.join(temp_dir, "empty.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("")
    docs = load_document(file_path, config)
    assert docs == []


def test_load_unsupported_file(config, temp_dir):
    file_path = os.path.join(temp_dir, "test.xyz")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("content")
    docs = load_document(file_path, config)
    assert docs == []


def test_load_mixed_batch(config, temp_dir):
    with open(os.path.join(temp_dir, "valid1.txt"), "w") as f:
        f.write("Test document one with meaningful content.")
    with open(os.path.join(temp_dir, "valid2.txt"), "w") as f:
        f.write("Test document two with more content here.")
    with open(os.path.join(temp_dir, "empty.txt"), "w") as f:
        f.write("")
    docs, _ = load_directory(temp_dir, config)
    assert len(docs) >= 2
