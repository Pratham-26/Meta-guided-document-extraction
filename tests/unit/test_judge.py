import pytest
from unittest.mock import MagicMock, patch

from src.schemas.evaluation import JudgeEvaluation, QualityTier, FieldDiff


class TestJudgeEvaluation:
    def test_low_tier(self):
        eval = JudgeEvaluation(
            quality_tier=QualityTier.LOW,
            feedback="Missing fields",
            field_diffs=[
                FieldDiff(
                    field="name", expected="Acme", actual=None, issue="Missing value"
                ),
            ],
            gold_standard_id="gs_001",
            confidence=0.2,
        )
        assert eval.quality_tier == QualityTier.LOW
        assert len(eval.field_diffs) == 1

    def test_high_tier(self):
        eval = JudgeEvaluation(
            quality_tier=QualityTier.HIGH,
            feedback="All correct",
            field_diffs=[],
            gold_standard_id="gs_001",
            confidence=0.98,
        )
        assert eval.quality_tier == QualityTier.HIGH
        assert eval.confidence == 0.98

    def test_quality_tier_values(self):
        assert QualityTier.LOW.value == "low"
        assert QualityTier.MEDIUM.value == "medium"
        assert QualityTier.HIGH.value == "high"
