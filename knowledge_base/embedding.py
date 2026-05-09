from .config import Config
from .ingestion import get_embeddings


def embed_chunks(chunks, config: Config):
    embeddings = get_embeddings(config)
    return embeddings.embed_documents(chunks)
