import dspy
import json
from pathlib import Path


class ScoutExplore(dspy.Signature):
    """Thoroughly explore a document to identify all extractable information and patterns.
    For PDF documents, analyze visual layout, tables, and spatial relationships.
    For text documents, analyze structure, key phrases, and semantic patterns."""

    document_content: str = dspy.InputField(
        desc="Full document content or description of visual layout"
    )
    extraction_schema: str = dspy.InputField(
        desc="JSON schema defining expected extraction fields"
    )
    extraction_instructions: str = dspy.InputField(
        desc="Extraction instructions for this category"
    )
    exploration: str = dspy.OutputField(
        desc="Detailed findings: document structure, key data points, patterns, edge cases"
    )
    extraction: str = dspy.OutputField(
        desc="Complete extraction as a JSON object matching the schema"
    )


class ScoutQuestionInference(dspy.Signature):
    """Given explorations of multiple documents from the same category, infer the essential
    questions that need to be answered during retrieval to support accurate extraction.
    Each question should target a specific field or piece of information."""

    explorations: str = dspy.InputField(
        desc="Exploration findings from multiple documents"
    )
    extraction_schema: str = dspy.InputField(
        desc="JSON schema defining expected extraction fields"
    )
    extraction_instructions: str = dspy.InputField(desc="Extraction instructions")
    questions_json: str = dspy.OutputField(
        desc='JSON array of objects: [{"text": "question text", "target_field": "field_name", "retrieval_priority": 1}]'
    )


class ScoutAgent:
    def __init__(self, lm: dspy.LM | None = None):
        if lm:
            dspy.configure(lm=lm)
        self.explore = dspy.Predict(ScoutExplore)
        self.infer_questions = dspy.Predict(ScoutQuestionInference)

    def explore_document(self, content: str, schema: dict, instructions: str) -> dict:
        result = self.explore(
            document_content=content,
            extraction_schema=json.dumps(schema, indent=2),
            extraction_instructions=instructions,
        )
        try:
            extraction = json.loads(result.extraction)
        except json.JSONDecodeError:
            extraction = {"raw": result.extraction}

        return {
            "exploration": result.exploration,
            "extraction": extraction,
        }

    def infer_questions_from_explorations(
        self,
        explorations: list[str],
        schema: dict,
        instructions: str,
    ) -> list[dict]:
        result = self.infer_questions(
            explorations=json.dumps(explorations, indent=2),
            extraction_schema=json.dumps(schema, indent=2),
            extraction_instructions=instructions,
        )
        try:
            questions = json.loads(result.questions_json)
        except json.JSONDecodeError:
            questions = []

        return questions
