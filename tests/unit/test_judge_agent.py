import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.judge.agent import JudgeAgent
from src.schemas.evaluation import JudgeEvaluation, QualityTier, FieldDiff


class TestJudgeAgentEvaluate:
    def test_returns_high_quality_evaluation(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.quality_tier = "high"
        fake_result.feedback = "All fields match the Gold Standard."
        fake_result.field_diffs = "[]"
        fake_result.confidence = "0.95"
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = JudgeAgent()
        result = agent.evaluate(
            extraction={"name": "Acme Corp", "amount": 5000.00},
            gold_standard={"name": "Acme Corp", "amount": 5000.00},
            schema={"type": "object"},
            gold_standard_id="gs_001",
        )

        assert isinstance(result, JudgeEvaluation)
        assert result.quality_tier == QualityTier.HIGH
        assert result.confidence == 0.95
        assert result.gold_standard_id == "gs_001"

    def test_returns_low_quality_with_field_diffs(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.quality_tier = "low"
        fake_result.feedback = "Name is missing."
        fake_result.field_diffs = json.dumps(
            [
                {
                    "field": "name",
                    "expected": "Acme Corp",
                    "actual": None,
                    "issue": "Missing value",
                }
            ]
        )
        fake_result.confidence = "0.3"
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = JudgeAgent()
        result = agent.evaluate(
            extraction={"amount": 5000.00},
            gold_standard={"name": "Acme Corp", "amount": 5000.00},
            schema={"type": "object"},
            gold_standard_id="gs_002",
        )

        assert result.quality_tier == QualityTier.LOW
        assert len(result.field_diffs) == 1
        assert result.field_diffs[0].field == "name"

    def test_handles_invalid_quality_tier(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.quality_tier = "unknown_tier"
        fake_result.feedback = "Some feedback"
        fake_result.field_diffs = "[]"
        fake_result.confidence = "0.5"
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = JudgeAgent()
        result = agent.evaluate(
            extraction={},
            gold_standard={},
            schema={},
            gold_standard_id="gs_001",
        )

        assert result.quality_tier == QualityTier.MEDIUM

    def test_handles_malformed_field_diffs(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.quality_tier = "medium"
        fake_result.feedback = "Partial match."
        fake_result.field_diffs = "not json"
        fake_result.confidence = "0.6"
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = JudgeAgent()
        result = agent.evaluate(
            extraction={"name": "Acme"},
            gold_standard={"name": "Acme Corp"},
            schema={},
            gold_standard_id="gs_001",
        )

        assert result.field_diffs == []

    def test_clamps_confidence_to_range(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.quality_tier = "high"
        fake_result.feedback = "Good"
        fake_result.field_diffs = "[]"
        fake_result.confidence = "1.5"
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = JudgeAgent()
        result = agent.evaluate(
            extraction={}, gold_standard={}, schema={}, gold_standard_id="gs_001"
        )

        assert result.confidence == 1.0

    def test_handles_non_numeric_confidence(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.quality_tier = "medium"
        fake_result.feedback = "Ok"
        fake_result.field_diffs = "[]"
        fake_result.confidence = "not a number"
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = JudgeAgent()
        result = agent.evaluate(
            extraction={}, gold_standard={}, schema={}, gold_standard_id="gs_001"
        )

        assert result.confidence == 0.5
