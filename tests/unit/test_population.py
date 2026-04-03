import json
from unittest.mock import patch

import pytest

from src.optimization.population import (
    PromptCandidate,
    save_candidate,
    load_candidate,
    list_candidates,
    save_current_prompt,
    load_current_prompt,
    pareto_select,
    _next_candidate_id,
)


class TestPromptCandidate:
    def test_creation(self):
        c = PromptCandidate(
            candidate_id="candidate_001",
            generation=1,
            created_at="2025-01-01T00:00:00+00:00",
            instructions="Extract name.",
        )
        assert c.candidate_id == "candidate_001"
        assert c.generation == 1
        assert c.fitness_scores == {}
        assert c.parent_ids == []

    def test_with_fitness_scores(self):
        c = PromptCandidate(
            candidate_id="candidate_002",
            generation=2,
            created_at="2025-01-01T00:00:00+00:00",
            instructions="Extract name.",
            fitness_scores={"overall_accuracy": 0.8, "high_count": 3},
        )
        assert c.fitness_scores["overall_accuracy"] == 0.8


class TestCandidateCRUD:
    def test_save_and_load(self, tmp_category_dir):
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_cat")

        c = PromptCandidate(
            candidate_id="candidate_001",
            generation=1,
            created_at="2025-01-01T00:00:00+00:00",
            instructions="Extract name.",
        )
        save_candidate("test_cat", c)

        loaded = load_candidate("test_cat", "candidate_001")
        assert loaded is not None
        assert loaded.candidate_id == "candidate_001"
        assert loaded.instructions == "Extract name."

    def test_load_nonexistent(self, tmp_category_dir):
        result = load_candidate("nonexistent", "candidate_999")
        assert result is None

    def test_list_candidates(self, tmp_category_dir):
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_cat")

        for i in range(3):
            c = PromptCandidate(
                candidate_id=f"candidate_{i + 1:03d}",
                generation=i + 1,
                created_at="2025-01-01T00:00:00+00:00",
                instructions=f"Instructions v{i + 1}",
            )
            save_candidate("test_cat", c)

        candidates = list_candidates("test_cat")
        assert len(candidates) == 3
        assert candidates[0].candidate_id == "candidate_001"

    def test_list_empty_category(self, tmp_category_dir):
        candidates = list_candidates("nonexistent")
        assert candidates == []


class TestCurrentPrompt:
    def test_save_and_load(self, tmp_category_dir):
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_cat")

        c = PromptCandidate(
            candidate_id="candidate_005",
            generation=3,
            created_at="2025-01-01T00:00:00+00:00",
            instructions="Best instructions.",
        )
        save_current_prompt("test_cat", c)

        loaded = load_current_prompt("test_cat")
        assert loaded is not None
        assert loaded.instructions == "Best instructions."

    def test_load_nonexistent(self, tmp_category_dir):
        result = load_current_prompt("nonexistent")
        assert result is None


class TestNextCandidateId:
    def test_first_candidate(self, tmp_category_dir):
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_cat")
        assert _next_candidate_id("test_cat") == "candidate_001"

    def test_after_existing(self, tmp_category_dir):
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_cat")

        c = PromptCandidate(
            candidate_id="candidate_001",
            generation=1,
            created_at="2025-01-01T00:00:00+00:00",
            instructions="Instructions.",
        )
        save_candidate("test_cat", c)

        assert _next_candidate_id("test_cat") == "candidate_002"


class TestParetoSelect:
    def test_single_candidate(self):
        c = PromptCandidate(
            candidate_id="c1",
            generation=1,
            created_at="2025-01-01T00:00:00+00:00",
            instructions="i1",
            fitness_scores={"accuracy": 0.5},
        )
        front = pareto_select([c])
        assert len(front) == 1

    def test_empty_list(self):
        assert pareto_select([]) == []

    def test_dominant_candidate_selected(self):
        c1 = PromptCandidate(
            candidate_id="c1",
            generation=1,
            created_at="2025-01-01T00:00:00+00:00",
            instructions="i1",
            fitness_scores={"accuracy": 0.9, "speed": 0.8},
        )
        c2 = PromptCandidate(
            candidate_id="c2",
            generation=1,
            created_at="2025-01-01T00:00:00+00:00",
            instructions="i2",
            fitness_scores={"accuracy": 0.5, "speed": 0.3},
        )
        front = pareto_select([c1, c2])
        assert len(front) == 1
        assert front[0].candidate_id == "c1"

    def test_no_dominance_both_in_front(self):
        c1 = PromptCandidate(
            candidate_id="c1",
            generation=1,
            created_at="2025-01-01T00:00:00+00:00",
            instructions="i1",
            fitness_scores={"accuracy": 0.9, "speed": 0.3},
        )
        c2 = PromptCandidate(
            candidate_id="c2",
            generation=1,
            created_at="2025-01-01T00:00:00+00:00",
            instructions="i2",
            fitness_scores={"accuracy": 0.3, "speed": 0.9},
        )
        front = pareto_select([c1, c2])
        assert len(front) == 2

    def test_empty_fitness_no_dominance(self):
        c1 = PromptCandidate(
            candidate_id="c1",
            generation=1,
            created_at="2025-01-01T00:00:00+00:00",
            instructions="i1",
        )
        c2 = PromptCandidate(
            candidate_id="c2",
            generation=1,
            created_at="2025-01-01T00:00:00+00:00",
            instructions="i2",
        )
        front = pareto_select([c1, c2])
        assert len(front) == 2
