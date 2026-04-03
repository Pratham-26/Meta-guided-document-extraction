import base64
import io

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


class ScoutExploreVision(dspy.Signature):
    """Thoroughly explore document page images to identify all extractable information
    and patterns. Analyze visual layout, tables, spatial relationships, fonts, and
    formatting to extract structured data."""

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


def _encode_images(images: list) -> list[dict]:
    import PIL.Image

    encoded = []
    for img in images:
        buf = io.BytesIO()
        pil_img = img if isinstance(img, PIL.Image.Image) else PIL.Image.open(img)
        pil_img.save(buf, format="PNG")
        encoded.append(
            {"type": "image", "image": base64.b64encode(buf.getvalue()).decode("utf-8")}
        )
    return encoded


class ScoutAgent:
    def __init__(
        self,
        lm: dspy.LM | None = None,
        vision_lm: dspy.LM | None = None,
    ):
        self._lm = lm
        self._vision_lm = vision_lm
        self.explore = dspy.RLM(ScoutExplore)
        self.explore_vision = dspy.RLM(ScoutExploreVision)
        self.infer_questions = dspy.Predict(ScoutQuestionInference)

    def _configure_lm(self, lm: dspy.LM | None):
        if lm is not None:
            dspy.configure(lm=lm)

    def explore_document(
        self,
        content: str,
        schema: dict,
        instructions: str,
        images: list | None = None,
    ) -> dict:
        schema_str = json.dumps(schema, indent=2)

        if images:
            encoded_images = _encode_images(images)
            prev_lm = self._lm
            self._configure_lm(self._vision_lm)

            try:
                result = self.explore_vision(
                    extraction_schema=schema_str,
                    extraction_instructions=instructions,
                )
            finally:
                self._configure_lm(prev_lm)
        else:
            self._configure_lm(self._lm)
            result = self.explore(
                document_content=content,
                extraction_schema=schema_str,
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
        self._configure_lm(self._lm)
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
