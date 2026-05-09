import logging
from typing import List

from langchain_core.documents import Document

from .config import Config
from .llm import get_llm

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """You are a research assistant answering questions based ONLY on the provided context.
If the context doesn't contain the answer, say "No relevant documents found in your knowledge base." Do NOT make up information.

Rules:
- Cite sources inline using numbered references: [1], [2], etc.
- If you use information from a source, reference it.
- Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""


def _format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i+1}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


def generate(question: str, context_docs: List[Document], config: Config) -> str:
    if not context_docs:
        return "No relevant documents found in your knowledge base."

    context = _format_context(context_docs)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    if config.debug:
        logger.info("Prompt length: %d chars, context from %d docs", len(prompt), len(context_docs))

    try:
        llm = get_llm(config)
        response = llm.invoke(prompt)
        return response.strip() if isinstance(response, str) else response
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            logger.error(
                "Response timed out after %ds. Try a smaller model (KB_MODEL=qwen2.5:3b) "
                "or shorter documents.", config.timeout
            )
            return "Response timed out. Try rephrasing your question or using a smaller model."
        if "connection" in error_msg.lower() or "connect" in error_msg.lower():
            logger.error(
                "Ollama is not running. Start it with: ollama serve"
            )
            return "Cannot connect to Ollama. Is it running? Start with: ollama serve"
        logger.error("LLM generation failed: %s", e)
        return f"Error generating answer: {error_msg[:200]}"
