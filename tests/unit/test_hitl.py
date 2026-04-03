from src.orchestration.state import PipelineState
from src.orchestration.hitl import present_for_review, apply_human_corrections


class TestPresentForReview:
    def test_returns_summary_with_all_fields(self):
        state: PipelineState = {
            "category_name": "leases",
            "questions": ["Who is the landlord?", "What is the rent?"],
            "extraction": {"name": "Acme Corp", "amount": 5000.0},
        }

        result = present_for_review(state)

        assert set(result.keys()) == {"category", "questions", "extraction"}
        assert result["category"] == "leases"
        assert result["questions"] == ["Who is the landlord?", "What is the rent?"]
        assert result["extraction"]["name"] == "Acme Corp"

    def test_handles_missing_optional_fields(self):
        state: PipelineState = {"category_name": "invoices"}

        result = present_for_review(state)

        assert result["category"] == "invoices"
        assert result["questions"] == []
        assert result["extraction"] is None

    def test_preserves_state(self):
        state: PipelineState = {
            "category_name": "leases",
            "questions": ["Q1"],
            "extraction": {"name": "Test"},
        }
        original = dict(state)

        present_for_review(state)

        assert state == original


class TestApplyHumanCorrections:
    def test_updates_extraction(self):
        state: PipelineState = {
            "category_name": "leases",
            "extraction": {"name": "Wrong Name", "amount": 1000},
        }
        corrections = {"extraction": {"name": "Correct Name", "amount": 5000}}

        result = apply_human_corrections(state, corrections)

        assert result["extraction"]["name"] == "Correct Name"
        assert result["extraction"]["amount"] == 5000

    def test_updates_questions(self):
        state: PipelineState = {
            "category_name": "leases",
            "questions": ["Old question"],
        }
        corrections = {"questions": ["New question 1", "New question 2"]}

        result = apply_human_corrections(state, corrections)

        assert result["questions"] == ["New question 1", "New question 2"]

    def test_updates_both(self):
        state: PipelineState = {
            "category_name": "leases",
            "extraction": {"name": "Old"},
            "questions": ["Old Q"],
        }
        corrections = {
            "extraction": {"name": "New"},
            "questions": ["New Q"],
        }

        result = apply_human_corrections(state, corrections)

        assert result["extraction"]["name"] == "New"
        assert result["questions"] == ["New Q"]

    def test_preserves_unchanged_fields(self):
        state: PipelineState = {
            "category_name": "leases",
            "extraction": {"name": "Keep", "amount": 5000},
            "questions": ["Keep Q"],
        }
        corrections = {"extraction": {"name": "Changed"}}

        result = apply_human_corrections(state, corrections)

        assert result["category_name"] == "leases"
        assert result["questions"] == ["Keep Q"]
        assert result["extraction"]["name"] == "Changed"

    def test_empty_corrections_returns_unchanged_state(self):
        state: PipelineState = {
            "category_name": "leases",
            "extraction": {"name": "Acme"},
            "questions": ["Q1"],
        }

        result = apply_human_corrections(state, {})

        assert result == state
