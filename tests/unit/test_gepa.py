import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.optimization.gepa import run_gepa_cycle
from src.optimization.population import PromptCandidate, save_candidate


def _write_category_config(tmp_path, sample_category_config, category="test_category"):
    configs_dir = tmp_path / "configs"
    config_path = configs_dir / "categories" / f"{category}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(sample_category_config.model_dump_json(indent=2))
    return configs_dir


def _write_gold_standards(category, count=2):
    from src.storage.fs_store import save_gold_standard
    from src.schemas.gold_standard import GoldStandard

    gses = []
    for i in range(count):
        gs = GoldStandard(
            id=f"gs_{i + 1:03d}",
            category=category,
            source_document_uri=Path(f"sources/doc_{i + 1}.pdf"),
            extraction={"name": f"Entity {i + 1}", "amount": 1000.0 * (i + 1)},
            approved_by="scout",
            created_at=datetime.now(timezone.utc),
        )
        save_gold_standard(category, gs)
        gses.append(gs)
    return gses


class TestRunGepaCycle:
    def test_no_gold_standards_returns_error(
        self, tmp_category_dir, sample_category_config
    ):
        from src.config import settings
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_category")
        configs_dir = _write_category_config(tmp_category_dir, sample_category_config)

        with patch.object(settings, "configs_dir", configs_dir):
            result = run_gepa_cycle("test_category")

        assert "error" in result
        assert "No Gold Standards" in result["error"]

    def test_creates_initial_candidate_if_none_exists(
        self, tmp_category_dir, sample_category_config
    ):
        from src.config import settings
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_category")
        configs_dir = _write_category_config(tmp_category_dir, sample_category_config)
        _write_gold_standards("test_category")

        with patch.object(settings, "configs_dir", configs_dir):
            with (
                patch("src.optimization.gepa.get_lm") as mock_get_lm,
                patch(
                    "src.optimization.reflector.Reflector.analyze",
                    return_value={
                        "diagnosis": "Missing instruction.",
                        "suggested_fixes": "Add instruction.",
                    },
                ),
                patch(
                    "src.optimization.reflector.PromptMutator.mutate",
                    return_value={
                        "revised_instructions": "Improved instructions.",
                        "mutation_rationale": "Better clarity.",
                    },
                ),
                patch("src.optimization.gepa.validate_candidate") as mock_validate,
                patch("src.optimization.gepa.log_trace"),
            ):
                mock_get_lm.return_value = MagicMock()
                mock_validate.return_value = {
                    "total": 2,
                    "low": 0,
                    "medium": 1,
                    "high": 1,
                    "accuracy": 0.5,
                    "details": [],
                }

                result = run_gepa_cycle(
                    "test_category", generations=1, population_size=1
                )

                assert "error" not in result
                assert result["generations_run"] == 1

        from src.optimization.population import load_current_prompt

        current = load_current_prompt("test_category")
        assert current is not None

    def test_runs_multiple_generations(self, tmp_category_dir, sample_category_config):
        from src.config import settings
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_category")
        configs_dir = _write_category_config(tmp_category_dir, sample_category_config)
        _write_gold_standards("test_category")

        initial = PromptCandidate(
            candidate_id="candidate_000",
            generation=0,
            created_at=datetime.now(timezone.utc).isoformat(),
            instructions="Extract name and amount.",
        )
        save_candidate("test_category", initial)

        with patch.object(settings, "configs_dir", configs_dir):
            with (
                patch("src.optimization.gepa.get_lm") as mock_get_lm,
                patch(
                    "src.optimization.reflector.Reflector.analyze",
                    return_value={
                        "diagnosis": "diag",
                        "suggested_fixes": "fixes",
                    },
                ),
                patch(
                    "src.optimization.reflector.PromptMutator.mutate",
                    return_value={
                        "revised_instructions": "Revised instructions.",
                        "mutation_rationale": "Rationale.",
                    },
                ),
                patch("src.optimization.gepa.validate_candidate") as mock_validate,
                patch("src.optimization.gepa.log_trace") as mock_log,
            ):
                mock_get_lm.return_value = MagicMock()
                mock_validate.return_value = {
                    "total": 2,
                    "low": 0,
                    "medium": 1,
                    "high": 1,
                    "accuracy": 0.5,
                    "details": [],
                }

                result = run_gepa_cycle(
                    "test_category", generations=2, population_size=1
                )

                assert result["generations_run"] == 2
                assert mock_log.call_count == 2

    def test_skips_generation_when_no_low_medium_samples(
        self, tmp_category_dir, sample_category_config
    ):
        from src.config import settings
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_category")
        configs_dir = _write_category_config(tmp_category_dir, sample_category_config)
        _write_gold_standards("test_category")

        with patch.object(settings, "configs_dir", configs_dir):
            with (
                patch("src.optimization.gepa.get_lm") as mock_get_lm,
                patch(
                    "src.optimization.gepa._select_low_medium_samples"
                ) as mock_select,
                patch(
                    "src.optimization.reflector.Reflector.analyze",
                    return_value={"diagnosis": "d", "suggested_fixes": "f"},
                ),
                patch(
                    "src.optimization.reflector.PromptMutator.mutate",
                    return_value={
                        "revised_instructions": "r",
                        "mutation_rationale": "m",
                    },
                ),
                patch("src.optimization.gepa.validate_candidate") as mock_validate,
                patch("src.optimization.gepa.log_trace") as mock_log,
            ):
                mock_get_lm.return_value = MagicMock()
                mock_select.return_value = []
                mock_validate.return_value = {
                    "total": 2,
                    "low": 0,
                    "medium": 1,
                    "high": 1,
                    "accuracy": 0.5,
                    "details": [],
                }

                result = run_gepa_cycle(
                    "test_category", generations=2, population_size=1
                )

                assert result["generations_run"] == 2
