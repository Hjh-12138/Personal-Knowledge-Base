import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import Config

logger = logging.getLogger(__name__)

LOADER_MAP = {
    ".pdf": (PyPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".md": (TextLoader, {"encoding": "utf-8"}),
}


def _validate_document(doc: Document, source_path: str) -> Optional[Document]:
    """Pre-ingestion validation gate. Returns doc if valid, None if should skip."""
    if doc.page_content is None or len(doc.page_content.strip()) == 0:
        logger.warning("Skipping %s: no extractable text found (empty document)", source_path)
        return None

    try:
        doc.page_content.encode("utf-8")
    except UnicodeEncodeError:
        logger.warning("Skipping %s: binary content (misnamed file extension?)", source_path)
        return None

    return doc


def _make_chunk_id(doc_path: str, chunk_index: int) -> str:
    """Deterministic chunk ID for Chroma upsert."""
    raw = f"{doc_path}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def load_document(file_path: str, config: Config) -> List[Document]:
    """Load a single document with validation. Returns list of Documents or empty on skip."""
    ext = Path(file_path).suffix.lower()
    loader_cls, loader_kwargs = LOADER_MAP.get(ext, (None, None))

    if loader_cls is None:
        logger.warning("Skipping %s: unsupported file type (%s)", file_path, ext)
        return []

    try:
        loader = loader_cls(file_path, **loader_kwargs)
        docs = loader.load()
    except Exception as e:
        logger.error("Could not read %s — it may be a scanned image or encrypted %s. Error: %s",
                      file_path, ext.upper(), e)
        return []

    validated = []
    for doc in docs:
        doc.metadata["source"] = file_path
        valid = _validate_document(doc, file_path)
        if valid is not None:
            validated.append(valid)

    if not validated and docs:
        logger.warning("Skipping %s: all pages/content empty after validation", file_path)

    return validated


def load_directory(directory: str, config: Config):
    """Load all supported documents from a directory with per-file error handling.

    Returns (docs, skipped_count).
    """
    supported_exts = list(LOADER_MAP.keys())
    all_docs = []
    skipped = 0
    loaded = 0

    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning("Directory %s does not exist. Creating it.", directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return [], 0

    for ext in supported_exts:
        for file_path in dir_path.glob(f"*{ext}"):
            docs = load_document(str(file_path), config)
            if docs:
                all_docs.extend(docs)
                loaded += 1
            else:
                skipped += 1
        for file_path in dir_path.glob(f"*{ext.upper()}"):
            docs = load_document(str(file_path), config)
            if docs:
                all_docs.extend(docs)
                loaded += 1
            else:
                skipped += 1

    if config.debug:
        logger.info("Loaded %d documents, skipped %d files", loaded, skipped)

    return all_docs, skipped


def chunk_documents(docs: List[Document], config: Config) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )
    return splitter.split_documents(docs)


def get_embeddings(config: Config):
    if config.debug:
        logger.info("Loading embedding model: %s (first run downloads ~80MB)", config.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"trust_remote_code": True},
    )


def ingest_directory(directory: str, config: Config, collection_name: str = "knowledge_base"):
    """Full ingestion pipeline: load -> validate -> chunk -> embed -> store.

    Returns (chunk_count, skipped_count).
    Uses deterministic chunk IDs for safe re-indexing (upsert without duplicates).
    """
    embeddings = get_embeddings(config)

    docs, skipped = load_directory(directory, config)
    if not docs:
        return 0, skipped

    chunks = chunk_documents(docs, config)
    if config.debug:
        logger.info("Loaded %d docs -> %d chunks, embedding...", len(docs), len(chunks))

    # Deterministic IDs: (doc_path, chunk_index) -> hash -> Chroma upsert
    ids = [
        _make_chunk_id(chunk.metadata.get("source", f"unknown_{i}"), i)
        for i, chunk in enumerate(chunks)
    ]

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=config.chroma_persist_dir,
    )
    vector_store.add_documents(documents=chunks, ids=ids)

    # 索引后刷新检索缓存，确保后续搜索使用最新数据
    from .retrieval import invalidate_cache
    invalidate_cache()

    if config.debug:
        logger.info("Stored %d chunks in Chroma (collection: %s)", len(chunks), collection_name)

    return len(chunks), skipped
