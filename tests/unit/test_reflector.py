from unittest.mock import MagicMock, patch

import pytest

from src.optimization.reflector import Reflector, PromptMutator


class TestReflectorAnalyze:
    def test_returns_diagnosis_and_fixes(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.diagnosis = "Missing instruction to look at signature block."
        fake_result.suggested_fixes = (
            "Add instruction to check signature block for names."
        )
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        reflector = Reflector()
        result = reflector.analyze(
            extraction={"name": None, "amount": 5000},
            gold_standard={"name": "Acme Corp", "amount": 5000},
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            current_instructions="Extract name and amount.",
            field_feedback="name field is missing",
        )

        assert "diagnosis" in result
        assert "suggested_fixes" in result
        assert "signature block" in result["diagnosis"]

    def test_passes_schema_as_extraction_schema(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.diagnosis = "diagnosis"
        fake_result.suggested_fixes = "fixes"
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        reflector = Reflector()
        schema = {"type": "object"}
        reflector.analyze(
            extraction={},
            gold_standard={},
            schema=schema,
            current_instructions="instructions",
            field_feedback="feedback",
        )

        call_kwargs = mock_dspy_predict.return_value.call_args[1]
        import json

        assert call_kwargs["extraction_schema"] == json.dumps(schema, indent=2)


class TestPromptMutatorMutate:
    def test_returns_revised_instructions(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.revised_instructions = "Extract name from the signature block. Extract amount from the rent section."
        fake_result.mutation_rationale = (
            "Added specific location hints for name extraction."
        )
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        mutator = PromptMutator()
        result = mutator.mutate(
            current_instructions="Extract name and amount.",
            diagnosis="Name not found in header.",
            suggested_fixes="Look in signature block for name.",
        )

        assert "revised_instructions" in result
        assert "mutation_rationale" in result
        assert "signature block" in result["revised_instructions"]

    def test_mutation_differs_from_current(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.revised_instructions = "New improved instructions."
        fake_result.mutation_rationale = "Improved clarity."
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        mutator = PromptMutator()
        result = mutator.mutate(
            current_instructions="Original instructions.",
            diagnosis="Too vague.",
            suggested_fixes="Be more specific.",
        )

        assert result["revised_instructions"] != "Original instructions."
