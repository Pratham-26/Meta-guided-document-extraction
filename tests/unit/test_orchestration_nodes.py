from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.orchestration.nodes import (
    check_context,
    resolve_config,
    load_questions,
    route_input,
    extract,
    judge,
    log_traces,
)
from src.orchestration.state import PipelineState
from src.schemas.document import DocumentInput, InputType
from src.schemas.evaluation import JudgeEvaluation, QualityTier
from src.retrieval.router import RetrievalRoute


class TestCheckContext:
    def test_no_context_returns_error(self, tmp_category_dir):
        state = {"category_name": "nonexistent"}
        result = check_context(state)
        assert "error" in result
        assert "No Scout context" in result["error"]

    def test_with_context_returns_state(self, bootstrapped_category):
        state = {"category_name": "test_category"}
        result = check_context(state)
        assert "error" not in result


class TestResolveConfig:
    def test_loads_category_config(self, tmp_category_dir, sample_category_config):
        from src.config import settings

        configs_dir = tmp_category_dir / "configs"
        config_path = configs_dir / "categories" / "test_category.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(sample_category_config.model_dump_json(indent=2))

        with patch.object(settings, "configs_dir", configs_dir):
            state = {"category_name": "test_category"}
            result = resolve_config(state)

        assert "schema" in result
        assert "instructions" in result
        assert result["instructions"] == "Extract name and amount."

    def test_missing_config_raises(self, tmp_category_dir):
        from src.config import settings

        with patch.object(settings, "configs_dir", tmp_category_dir / "configs"):
            state = {"category_name": "nonexistent"}
            with pytest.raises(FileNotFoundError):
                resolve_config(state)


class TestLoadQuestions:
    def test_loads_questions(self, bootstrapped_category):
        state = {"category_name": "test_category"}
        result = load_questions(state)
        assert "questions" in result
        assert len(result["questions"]) == 2

    def test_no_questions_returns_error(self, tmp_category_dir):
        state = {"category_name": "nonexistent"}
        result = load_questions(state)
        assert "error" in result


class TestRouteInput:
    def test_routes_pdf(self, sample_document_input):
        state = {"document": sample_document_input}
        result = route_input(state)
        assert result["retrieval_route"] == RetrievalRoute.COLPALI

    def test_no_document_returns_error(self):
        state = {}
        result = route_input(state)
        assert "error" in result


class TestExtract:
    def test_extracts_with_mocked_agent(
        self, bootstrapped_category, sample_category_config
    ):
        from src.config import settings

        configs_dir = bootstrapped_category / "configs"
        config_path = configs_dir / "categories" / "test_category.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(sample_category_config.model_dump_json(indent=2))

        state = {
            "category_name": "test_category",
            "retrieved_context": "Page 1: Lease agreement...",
            "schema": sample_category_config.expected_schema,
            "instructions": sample_category_config.extraction_instructions,
        }

        with patch.object(settings, "configs_dir", configs_dir):
            with patch("src.orchestration.nodes.get_lm") as mock_get_lm:
                mock_get_lm.return_value = MagicMock()
                with patch("src.agents.extractor.agent.ExtractorAgent") as mock_cls:
                    mock_agent = MagicMock()
                    mock_agent.run.return_value = {
                        "name": "Acme Corp",
                        "amount": 5000.00,
                    }
                    mock_cls.return_value = mock_agent

                    with patch(
                        "src.agents.extractor.few_shot.select_examples",
                        return_value=[],
                    ):
                        result = extract(state)

        assert result["extraction"]["name"] == "Acme Corp"
        assert len(result["trace_entries"]) == 1

    def test_skips_on_error(self):
        state = {"error": "something went wrong"}
        result = extract(state)
        assert "error" in state


class TestJudge:
    def test_judges_with_mocked_agent(
        self, bootstrapped_category, sample_category_config
    ):
        from src.config import settings

        configs_dir = bootstrapped_category / "configs"
        config_path = configs_dir / "categories" / "test_category.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(sample_category_config.model_dump_json(indent=2))

        state = {
            "category_name": "test_category",
            "extraction": {"name": "Acme Corp", "amount": 5000.00},
            "schema": sample_category_config.expected_schema,
        }

        eval_result = JudgeEvaluation(
            quality_tier=QualityTier.HIGH,
            feedback="All match.",
            field_diffs=[],
            gold_standard_id="gs_001",
            confidence=0.95,
        )

        with patch.object(settings, "configs_dir", configs_dir):
            with patch("src.orchestration.nodes.get_lm") as mock_get_lm:
                mock_get_lm.return_value = MagicMock()
                with patch("src.agents.judge.agent.JudgeAgent") as mock_cls:
                    mock_agent = MagicMock()
                    mock_agent.evaluate.return_value = eval_result
                    mock_cls.return_value = mock_agent

                    result = judge(state)

        assert result["judge_evaluation"].quality_tier == QualityTier.HIGH
        assert len(result["trace_entries"]) == 1

    def test_skips_on_error(self):
        state = {"error": "something went wrong"}
        result = judge(state)
        assert "error" in state

    def test_skips_when_no_gold_standards(self, tmp_category_dir):
        state = {
            "category_name": "empty_cat",
            "extraction": {"name": "Test"},
            "schema": {},
        }
        result = judge(state)
        assert result.get("judge_evaluation") is None


class TestLogTraces:
    def test_logs_all_traces(self):
        trace = MagicMock()
        state = {"trace_entries": [trace]}

        with patch("src.orchestration.nodes.log_trace") as mock_log:
            result = log_traces(state)
            mock_log.assert_called_once_with(trace)

    def test_handles_empty_traces(self):
        state = {"trace_entries": []}
        result = log_traces(state)
        assert "trace_entries" in result
