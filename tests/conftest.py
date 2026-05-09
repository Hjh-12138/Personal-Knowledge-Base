import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _clear_retrieval_cache():
    """每个测试前清空检索缓存，确保测试隔离。"""
    from knowledge_base.retrieval import invalidate_cache
    invalidate_cache()
