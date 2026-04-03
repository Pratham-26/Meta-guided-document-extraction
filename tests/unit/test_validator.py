from unittest.mock import patch, MagicMock

import pytest

from src.schemas.evaluation import JudgeEvaluation, QualityTier


class TestValidateCandidate:
    def test_no_gold_standards_returns_error(self, tmp_category_dir):
        with patch("src.optimization.validator.list_gold_standards", return_value=[]):
            from src.optimization.validator import validate_candidate

            result = validate_candidate(
                category="empty_cat",
                modality="pdf",
                instructions="Extract data.",
                schema={},
            )
        assert "error" in result
        assert "No Gold Standards" in result["error"]

    def test_validates_samples_with_mocked_agents(self, bootstrapped_category):
        with patch("src.optimization.validator.list_gold_standards") as mock_list:
            mock_list.return_value = bootstrapped_category
            mock_list.return_value = [
                MagicMock(
                    id="gs_001",
                    extraction={"name": "Entity 1", "amount": 1000.0},
                ),
                MagicMock(
                    id="gs_002",
                    extraction={"name": "Entity 2", "amount": 2000.0},
                ),
            ]

            eval_high = JudgeEvaluation(
                quality_tier=QualityTier.HIGH,
                feedback="Perfect match.",
                field_diffs=[],
                gold_standard_id="gs_001",
                confidence=0.95,
            )
            eval_medium = JudgeEvaluation(
                quality_tier=QualityTier.MEDIUM,
                feedback="Close but off on amount.",
                field_diffs=[],
                gold_standard_id="gs_002",
                confidence=0.6,
            )

            with patch("src.optimization.validator.get_lm") as mock_get_lm:
                mock_get_lm.return_value = MagicMock()
                with patch("src.optimization.validator.ExtractorAgent") as mock_ext_cls:
                    mock_ext = MagicMock()
                    mock_ext.run.side_effect = [
                        {"name": "Entity 1", "amount": 1000.0},
                        {"name": "Entity 2", "amount": 2500.0},
                    ]
                    mock_ext_cls.return_value = mock_ext

                    with patch(
                        "src.optimization.validator.JudgeAgent"
                    ) as mock_judge_cls:
                        mock_judge = MagicMock()
                        mock_judge.evaluate.side_effect = [eval_high, eval_medium]
                        mock_judge_cls.return_value = mock_judge

                        from src.optimization.validator import validate_candidate

                        result = validate_candidate(
                            category="test_category",
                            modality="pdf",
                            instructions="Extract name and amount.",
                            schema={"type": "object"},
                            sample_size=2,
                        )

        assert result["total"] == 2
        assert result["high"] == 1
        assert result["medium"] == 1
        assert result["low"] == 0
        assert result["accuracy"] == 0.5
        assert len(result["details"]) == 2
        assert result["details"][0]["quality_tier"] == "high"
        assert result["details"][1]["quality_tier"] == "medium"

    def test_sample_size_limits_results(self, tmp_category_dir):
        fake_gs_list = [
            MagicMock(id=f"gs_{i:03d}", extraction={"val": i}) for i in range(10)
        ]

        eval_result = JudgeEvaluation(
            quality_tier=QualityTier.HIGH,
            feedback="Good.",
            field_diffs=[],
            gold_standard_id="gs_000",
            confidence=0.9,
        )

        with patch(
            "src.optimization.validator.list_gold_standards",
            return_value=fake_gs_list,
        ):
            with patch("src.optimization.validator.get_lm") as mock_get_lm:
                mock_get_lm.return_value = MagicMock()
                with patch("src.optimization.validator.ExtractorAgent") as mock_ext_cls:
                    mock_ext = MagicMock()
                    mock_ext.run.return_value = {"val": 0}
                    mock_ext_cls.return_value = mock_ext

                    with patch(
                        "src.optimization.validator.JudgeAgent"
                    ) as mock_judge_cls:
                        mock_judge = MagicMock()
                        mock_judge.evaluate.return_value = eval_result
                        mock_judge_cls.return_value = mock_judge

                        from src.optimization.validator import validate_candidate

                        result = validate_candidate(
                            category="test_category",
                            modality="pdf",
                            instructions="Extract.",
                            schema={},
                            sample_size=3,
                        )

        assert result["total"] == 3
        mock_ext.run.assert_called()
        mock_judge.evaluate.assert_called()

    def test_accuracy_zero_when_no_high(self, tmp_category_dir):
        fake_gs = [MagicMock(id="gs_001", extraction={})]

        eval_result = JudgeEvaluation(
            quality_tier=QualityTier.LOW,
            feedback="Bad.",
            field_diffs=[],
            gold_standard_id="gs_001",
            confidence=0.1,
        )

        with patch(
            "src.optimization.validator.list_gold_standards",
            return_value=fake_gs,
        ):
            with patch("src.optimization.validator.get_lm") as mock_get_lm:
                mock_get_lm.return_value = MagicMock()
                with patch("src.optimization.validator.ExtractorAgent") as mock_ext_cls:
                    mock_ext = MagicMock()
                    mock_ext.run.return_value = {}
                    mock_ext_cls.return_value = mock_ext

                    with patch(
                        "src.optimization.validator.JudgeAgent"
                    ) as mock_judge_cls:
                        mock_judge = MagicMock()
                        mock_judge.evaluate.return_value = eval_result
                        mock_judge_cls.return_value = mock_judge

                        from src.optimization.validator import validate_candidate

                        result = validate_candidate(
                            category="test_category",
                            modality="pdf",
                            instructions="Extract.",
                            schema={},
                        )

        assert result["accuracy"] == 0.0
        assert result["low"] == 1
