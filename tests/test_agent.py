import pytest
from unittest.mock import patch, MagicMock

from knowledge_base.config import Config
from knowledge_base.agent import (
    decompose_query,
    web_search_tool,
    research_sub_question,
    synthesize_results,
    research,
)


@pytest.fixture
def config():
    return Config(debug=False)


class TestDecomposeQuery:
    def test_decompose_returns_list(self, config):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "1. What is the history of X?\n"
            "2. How does X work in practice?\n"
            "3. What are the limitations of X?"
        )

        with patch("knowledge_base.agent.get_llm", return_value=mock_llm):
            result = decompose_query("Explain everything about X", config)
            assert len(result) == 3
            assert "history of X" in result[0]

    def test_decompose_single_question_fallback(self, config):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "1. Explain X in detail with examples and context."

        with patch("knowledge_base.agent.get_llm", return_value=mock_llm):
            result = decompose_query("Explain X", config)
            assert len(result) == 1

    def test_decompose_handles_llm_failure(self, config):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Ollama down")

        with patch("knowledge_base.agent.get_llm", return_value=mock_llm):
            result = decompose_query("Complex question about X", config)
            assert len(result) == 1
            assert result[0] == "Complex question about X"

    def test_decompose_filters_short_lines(self, config):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "1. X\n"   # too short (<10 chars)
            "2. What is the comprehensive history of machine learning?\n"
            "3. Y\n"   # too short
            "4. How does deep learning differ from traditional ML?"
        )

        with patch("knowledge_base.agent.get_llm", return_value=mock_llm):
            result = decompose_query("Tell me about ML", config)
            assert len(result) == 2


class TestWebSearchTool:
    def test_web_search_returns_formatted_results(self, config):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
        ]

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs):
            result = web_search_tool("test query", config)
            assert "Result 1" in result
            assert "Result 2" in result
            assert "https://example.com/1" in result

    def test_web_search_handles_failure(self, config):
        with patch(
            "duckduckgo_search.DDGS",
            side_effect=Exception("Network error"),
        ):
            result = web_search_tool("test query", config)
            assert "unavailable" in result.lower()

    def test_web_search_no_results(self, config):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = []

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs):
            result = web_search_tool("xyzzy123", config)
            assert "No web results" in result


class TestResearchSubQuestion:
    def test_research_with_kb_results(self, config):
        mock_docs = [MagicMock()]
        mock_docs[0].metadata = {"source": "test.pdf"}

        with patch("knowledge_base.agent.search", return_value=mock_docs), \
             patch("knowledge_base.generation.generate", return_value="Answer from KB"):
            result = research_sub_question("test question", config)
            assert result["answer"] == "Answer from KB"
            assert len(result["kb_results"]) == 1

    def test_research_no_results(self, config):
        with patch("knowledge_base.agent.search", return_value=[]):
            result = research_sub_question("unknown topic", config)
            assert "No relevant documents" in result["answer"]


class TestSynthesizeResults:
    def test_synthesize_multiple_results(self, config):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Synthesized answer covering all aspects."

        sub_results = [
            {"question": "Q1", "answer": "Answer 1", "web_results": ""},
            {"question": "Q2", "answer": "Answer 2", "web_results": ""},
        ]

        with patch("knowledge_base.agent.get_llm", return_value=mock_llm):
            result = synthesize_results("Main question", sub_results, config)
            assert result == "Synthesized answer covering all aspects."

    def test_synthesize_single_result_returns_directly(self, config):
        sub_results = [{"question": "Q1", "answer": "Direct answer"}]
        result = synthesize_results("Main question", sub_results, config)
        assert result == "Direct answer"

    def test_synthesize_handles_llm_failure(self, config):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Timeout")

        sub_results = [
            {"question": "Q1", "answer": "Answer 1"},
            {"question": "Q2", "answer": "Answer 2"},
        ]

        with patch("knowledge_base.agent.get_llm", return_value=mock_llm):
            result = synthesize_results("Main question", sub_results, config)
            assert "Answer 1" in result
            assert "Answer 2" in result
            assert "Research Results" in result


class TestResearchPipeline:
    def test_research_full_pipeline(self, config):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "1. Sub-question A with enough length?\n2. Sub-question B with enough length?",
            "Synthesized final answer.",
        ]
        mock_docs = [MagicMock()]
        mock_docs[0].metadata = {"source": "doc.pdf"}

        with patch("knowledge_base.agent.get_llm", return_value=mock_llm), \
             patch("knowledge_base.agent.search", return_value=mock_docs), \
             patch("knowledge_base.generation.generate", return_value="Sub-answer"):
            result = research("Complex research question", config)
            assert result["question"] == "Complex research question"
            assert len(result["sub_questions"]) == 2
            assert len(result["sub_results"]) == 2
            assert result["answer"] == "Synthesized final answer."

    def test_research_logs_to_tracker(self, config):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "1. Sub-question one with enough content?\n2. Sub-question two with enough content?",
            "Synthesized answer.",
        ]
        mock_docs = [MagicMock()]
        mock_docs[0].metadata = {"source": "source.pdf"}

        mock_tracker = MagicMock()

        with patch("knowledge_base.agent.get_llm", return_value=mock_llm), \
             patch("knowledge_base.agent.search", return_value=mock_docs), \
             patch("knowledge_base.generation.generate", return_value="Sub-answer"):
            result = research(
                "Test question", config,
                tracker=mock_tracker,
                session_id="test_session",
                topic="Science",
            )
            mock_tracker.log_qa.assert_called_once()
            call_kwargs = mock_tracker.log_qa.call_args[1]
            assert call_kwargs["question"] == "Test question"
            assert call_kwargs["session_id"] == "test_session"

    def test_research_tracker_failure_not_propagated(self, config):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "1. Sub-question one with extra text for length?\n2. Sub-question two also with more text?",
            "Final answer.",
        ]
        mock_docs = [MagicMock()]
        mock_docs[0].metadata = {"source": "doc.pdf"}

        mock_tracker = MagicMock()
        mock_tracker.log_qa.side_effect = Exception("DB error")

        with patch("knowledge_base.agent.get_llm", return_value=mock_llm), \
             patch("knowledge_base.agent.search", return_value=mock_docs), \
             patch("knowledge_base.generation.generate", return_value="Sub-answer"):
            result = research("Question", config, tracker=mock_tracker)
            assert result["answer"] == "Final answer."

    def test_research_with_web_search(self, config):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "1. Sub-question alpha with enough text length here?\n2. Sub-question beta with enough text too?",
            "Synthesized with web results.",
        ]
        mock_docs = [MagicMock()]
        mock_docs[0].metadata = {"source": "doc.pdf"}

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Web", "href": "https://ex.com", "body": "Web result"}
        ]

        with patch("knowledge_base.agent.get_llm", return_value=mock_llm), \
             patch("knowledge_base.agent.search", return_value=mock_docs), \
             patch("knowledge_base.generation.generate", return_value="Sub-answer"), \
             patch("duckduckgo_search.DDGS", return_value=mock_ddgs):
            result = research("Question", config, use_web=True)
            assert result["sub_results"][0]["web_results"] != ""

    def test_decompose_empty_response(self, config):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ""

        with patch("knowledge_base.agent.get_llm", return_value=mock_llm):
            result = decompose_query("Question", config)
            assert result == ["Question"]
