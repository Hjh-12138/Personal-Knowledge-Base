import pytest
from unittest.mock import patch, MagicMock

from knowledge_base.config import Config
from knowledge_base.llm import get_llm, _StringWrapper

langchain_anthropic_available = False
try:
    import langchain_anthropic  # noqa: F401
    langchain_anthropic_available = True
except ImportError:
    pass


@pytest.fixture
def config():
    return Config(debug=False, timeout=30, temperature=0.3)


class TestGetLLM:
    def test_ollama_backend_returns_wrapper(self, config):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "response"

        with patch("langchain_ollama.OllamaLLM", return_value=mock_llm):
            wrapper = get_llm(config)
            result = wrapper.invoke("test prompt")
            assert result == "response"

    def test_vllm_backend(self, config):
        config.model_backend = "vllm"
        config.model_name = "mistral-7b"
        config.api_base_url = "http://10.0.0.5:8000/v1"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="vllm response")

        with patch("langchain_openai.ChatOpenAI", return_value=mock_llm):
            wrapper = get_llm(config)
            result = wrapper.invoke("test")
            assert result == "vllm response"

    def test_vllm_defaults_base_url(self, config):
        config.model_backend = "vllm"
        config.api_base_url = ""

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="ok")

        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_chat.return_value = mock_llm
            wrapper = get_llm(config)
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:8000/v1"
            assert call_kwargs["api_key"] == "not-needed"

    def test_openai_backend_uses_env_key(self, config):
        config.model_backend = "openai"
        config.model_name = "gpt-4o"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="openai response")

        with patch("langchain_openai.ChatOpenAI", return_value=mock_llm), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
            wrapper = get_llm(config)
            result = wrapper.invoke("test")
            assert result == "openai response"

    def test_openai_backend_with_config_key(self, config):
        config.model_backend = "openai"
        config.api_key = "sk-config-key"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="ok")

        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            # No env key set
            with patch.dict("os.environ", {}, clear=True):
                mock_chat.return_value = mock_llm
                get_llm(config)
                call_kwargs = mock_chat.call_args[1]
                assert call_kwargs["api_key"] == "sk-config-key"

    @pytest.mark.skipif(not langchain_anthropic_available, reason="langchain-anthropic not installed")
    def test_anthropic_backend(self, config):
        config.model_backend = "anthropic"
        config.model_name = "claude-sonnet-4-6"
        config.api_key = "sk-ant-test"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="anthropic response")

        with patch("langchain_anthropic.ChatAnthropic", return_value=mock_llm):
            wrapper = get_llm(config)
            result = wrapper.invoke("test")
            assert result == "anthropic response"

    def test_unknown_backend_raises(self, config):
        config.model_backend = "unknown_backend_xyz"
        with pytest.raises(ValueError, match="Unknown model_backend"):
            get_llm(config)

    def test_debug_logs_backend_info(self, config):
        config.debug = True
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "ok"

        with patch("langchain_ollama.OllamaLLM", return_value=mock_llm), \
             patch("knowledge_base.llm.logger") as mock_logger:
            get_llm(config)
            mock_logger.info.assert_called()


class TestStringWrapper:
    def test_wraps_string_result(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "  plain string  "
        wrapper = _StringWrapper(mock_llm)
        assert wrapper.invoke("prompt") == "plain string"

    def test_wraps_message_result(self):
        mock_llm = MagicMock()
        msg = MagicMock()
        msg.content = "  message content  "
        mock_llm.invoke.return_value = msg
        wrapper = _StringWrapper(mock_llm)
        assert wrapper.invoke("prompt") == "message content"

    def test_passes_prompt_to_inner_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "ok"
        wrapper = _StringWrapper(mock_llm)
        wrapper.invoke("my specific prompt")
        mock_llm.invoke.assert_called_once_with("my specific prompt")
