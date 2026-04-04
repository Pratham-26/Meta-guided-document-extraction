import base64
import io
import logging

import dspy
import json
from pathlib import Path

logger = logging.getLogger(__name__)


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


_CHARS_PER_TOKEN = 4
_RLM_TOKEN_THRESHOLD = 32_000


class ScoutAgent:
    def __init__(
        self,
        lm: dspy.LM | None = None,
        vision_lm: dspy.LM | None = None,
        sub_lm: dspy.LM | None = None,
        rlm_max_iterations: int = 15,
        rlm_max_llm_calls: int = 30,
        rlm_verbose: bool = False,
    ):
        self._lm = lm
        self._vision_lm = vision_lm
        self.explore_predict = dspy.Predict(ScoutExplore)
        self.explore_rlm = dspy.RLM(
            ScoutExplore,
            max_iterations=rlm_max_iterations,
            max_llm_calls=rlm_max_llm_calls,
            verbose=rlm_verbose,
            sub_lm=sub_lm,
        )
        self.explore_vision = dspy.Predict(ScoutExploreVision)
        self.infer_questions = dspy.Predict(ScoutQuestionInference)

    def _configure_lm(self, lm: dspy.LM | None):
        if lm is not None:
            dspy.configure(lm=lm)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text) // _CHARS_PER_TOKEN

    def _should_use_rlm(self, content: str, schema_str: str, instructions: str) -> bool:
        total_chars = len(content) + len(schema_str) + len(instructions)
        return total_chars // _CHARS_PER_TOKEN >= _RLM_TOKEN_THRESHOLD

    def explore_document(
        self,
        content: str,
        schema: dict,
        instructions: str,
        images: list | None = None,
    ) -> dict:
        schema_str = json.dumps(schema, indent=2)

        if images:
            _encode_images(images)
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

            if self._should_use_rlm(content, schema_str, instructions):
                logger.info("Using RLM for large document (%d chars)", len(content))
                result = self.explore_rlm(
                    document_content=content,
                    extraction_schema=schema_str,
                    extraction_instructions=instructions,
                )
                if result.trajectory:
                    logger.info(
                        "RLM exploration completed in %d iterations",
                        len(result.trajectory),
                    )
            else:
                result = self.explore_predict(
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
