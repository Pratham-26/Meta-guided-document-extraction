import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.extractor.agent import ExtractorAgent


class TestExtractorAgentRun:
    def test_returns_parsed_json_extraction(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.extraction = json.dumps({"name": "Acme Corp", "amount": 5000.00})
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = ExtractorAgent()
        result = agent.run(
            context="Lease agreement text...",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            instructions="Extract name and amount.",
        )

        assert result["name"] == "Acme Corp"
        assert result["amount"] == 5000.00

    def test_passes_few_shot_examples(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.extraction = '{"name": "Test"}'
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = ExtractorAgent()
        agent.run(
            context="text",
            schema={},
            instructions="Extract.",
            few_shot_examples=[{"source": "doc1.pdf", "extraction": {"name": "Acme"}}],
        )

        call_kwargs = mock_dspy_predict.return_value.call_args[1]
        assert "Acme" in call_kwargs["few_shot_examples"]

    def test_handles_malformed_json(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.extraction = "not valid json at all"
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = ExtractorAgent()
        result = agent.run("text", {}, "instructions")

        assert "_parse_error" in result
        assert result["raw"] == "not valid json at all"

    def test_handles_empty_few_shot_examples(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.extraction = '{"name": "Acme"}'
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = ExtractorAgent()
        result = agent.run("text", {}, "instructions", few_shot_examples=None)

        call_kwargs = mock_dspy_predict.return_value.call_args[1]
        assert call_kwargs["few_shot_examples"] == ""

    def test_passes_schema_and_instructions(self, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.extraction = "{}"
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = ExtractorAgent()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        instructions = "Extract the company name."
        agent.run("text", schema, instructions)

        call_kwargs = mock_dspy_predict.return_value.call_args[1]
        assert call_kwargs["extraction_schema"] == json.dumps(schema, indent=2)
        assert call_kwargs["extraction_instructions"] == instructions
