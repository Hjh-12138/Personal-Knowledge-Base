import logging
import threading
from typing import List, Tuple, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import Config
from .ingestion import get_embeddings

logger = logging.getLogger(__name__)

# 模块级缓存 — 避免每次搜索重新加载嵌入模型 (~80MB)
_embeddings = None
_vector_store = None
_cache_lock = threading.RLock()  # RLock: 同一线程可重入，避免 get_vector_store → _get_cached_embeddings 死锁


def _get_cached_embeddings(config: Config):
    global _embeddings
    if _embeddings is None:
        with _cache_lock:
            if _embeddings is None:
                _embeddings = get_embeddings(config)
    return _embeddings


def get_vector_store(config: Config, collection_name: str = "knowledge_base") -> Chroma:
    global _vector_store
    if _vector_store is None or _vector_store._collection_name != collection_name:
        with _cache_lock:
            if _vector_store is None or _vector_store._collection_name != collection_name:
                embeddings = _get_cached_embeddings(config)
                _vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=config.chroma_persist_dir,
                )
    return _vector_store


def invalidate_cache():
    """Force re-creation of embeddings and vector store (e.g. after re-index)."""
    global _embeddings, _vector_store
    with _cache_lock:
        _embeddings = None
        _vector_store = None


def search(query: str, config: Config, k: int = None, collection_name: str = "knowledge_base") -> List:
    k = k or config.top_k
    vector_store = get_vector_store(config, collection_name)

    try:
        results = vector_store.similarity_search(query, k=k)
        results = results[:k]
        if config.debug:
            logger.info("Query '%s' -> %d results", query[:80], len(results))
        return results
    except Exception as e:
        logger.error("Retrieval failed for query '%s': %s", query[:80], e)
        return []


def retrieve_with_scores(query: str, config: Config, k: int = None,
                         collection_name: str = "knowledge_base") -> List[Tuple]:
    k = k or config.top_k
    vector_store = get_vector_store(config, collection_name)
    try:
        return vector_store.similarity_search_with_relevance_scores(query, k=k)
    except Exception as e:
        logger.error("Retrieval with scores failed: %s", e)
        return []
