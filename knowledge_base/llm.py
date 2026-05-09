import logging
import os
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)


def get_llm(config: Config) -> Any:
    """Return a LangChain-compatible LLM instance.

    Supports four backends selected via ``config.model_backend``:

    =========  ====================  ==========================================
    Backend    LangChain class       Notes
    =========  ====================  ==========================================
    ollama     OllamaLLM             Local Ollama server (default)
    vllm       ChatOpenAI            OpenAI-compatible endpoint (localhost:8000)
    openai     ChatOpenAI            OpenAI / any OpenAI-compatible API
    anthropic  ChatAnthropic         Anthropic Messages API
    =========  ====================  ==========================================

    Returns an object whose ``invoke(prompt: str) -> str`` always returns a
    plain string, regardless of the underlying LangChain class.
    """
    backend = config.model_backend
    temperature = getattr(config, "temperature", 0.3)

    if backend == "ollama":
        from langchain_ollama import OllamaLLM

        llm = OllamaLLM(
            model=config.model_name,
            timeout=config.timeout,
            temperature=temperature,
        )
        if config.debug:
            logger.info("LLM backend: ollama (model=%s)", config.model_name)
        return _StringWrapper(llm)

    elif backend == "vllm":

        base = config.api_base_url or "http://localhost:8000/v1"
        key = config.api_key or "not-needed"

        llm = _create_openai_compatible(
            model=config.model_name,
            base_url=base,
            api_key=key,
            timeout=config.timeout,
            temperature=temperature,
        )
        if config.debug:
            logger.info("LLM backend: vllm (model=%s, base=%s)", config.model_name, base)
        return _StringWrapper(llm)

    elif backend == "openai":

        key = config.api_key or os.environ.get("OPENAI_API_KEY", "")
        base = config.api_base_url or None

        llm = _create_openai_compatible(
            model=config.model_name,
            base_url=base,
            api_key=key,
            timeout=config.timeout,
            temperature=temperature,
        )
        if config.debug:
            logger.info("LLM backend: openai (model=%s)", config.model_name)
        return _StringWrapper(llm)

    elif backend == "anthropic":

        key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        llm = _create_anthropic(
            model=config.model_name,
            api_key=key,
            timeout=config.timeout,
            temperature=temperature,
        )
        if config.debug:
            logger.info("LLM backend: anthropic (model=%s)", config.model_name)
        return _StringWrapper(llm)

    else:
        raise ValueError(
            f"Unknown model_backend '{backend}'. "
            "Valid options: ollama, vllm, openai, anthropic"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_openai_compatible(
    model: str,
    base_url: str | None,
    api_key: str,
    timeout: int,
    temperature: float,
):
    from langchain_openai import ChatOpenAI

    kwargs: dict = dict(
        model=model,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
    )
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def _create_anthropic(
    model: str,
    api_key: str,
    timeout: int,
    temperature: float,
):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
    )


class _StringWrapper:
    """Wraps any LangChain LLM/ChatModel so that ``invoke(prompt)`` always
    returns a plain ``str``.

    ``OllamaLLM.invoke()`` returns ``str`` directly.
    ``ChatOpenAI.invoke()`` / ``ChatAnthropic.invoke()`` returns
    ``AIMessage`` whose text is in ``.content``.
    """

    def __init__(self, llm):
        self._llm = llm

    def invoke(self, prompt: str) -> str:
        result = self._llm.invoke(prompt)
        if hasattr(result, "content"):
            return result.content.strip()
        return str(result).strip()
