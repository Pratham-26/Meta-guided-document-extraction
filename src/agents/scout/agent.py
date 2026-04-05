import base64
import io
import logging
import math

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


class ScoutMergeVisionExplorations(dspy.Signature):
    """Merge multiple page-chunk explorations and partial extractions from a
    multi-page PDF into a single, unified exploration summary and a complete
    extraction that covers all pages."""

    partial_explorations: str = dspy.InputField(
        desc="JSON array of exploration summaries from individual page chunks"
    )
    partial_extractions: str = dspy.InputField(
        desc="JSON array of partial extraction objects from individual page chunks"
    )
    extraction_schema: str = dspy.InputField(
        desc="JSON schema defining expected extraction fields"
    )
    extraction_instructions: str = dspy.InputField(
        desc="Extraction instructions for this category"
    )
    exploration: str = dspy.OutputField(
        desc="Unified exploration summary covering the entire document"
    )
    extraction: str = dspy.OutputField(
        desc="Complete, merged extraction as a JSON object matching the schema"
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


class ScoutDeduplicateQuestions(dspy.Signature):
    """Given multiple sets of inferred questions from different document batches,
    merge and deduplicate them into a single, canonical list.  Keep the most
    specific/descriptive version of each question and eliminate pure duplicates."""

    candidate_questions: str = dspy.InputField(
        desc="JSON array of all candidate question objects from multiple batches"
    )
    extraction_schema: str = dspy.InputField(
        desc="JSON schema defining expected extraction fields"
    )
    questions_json: str = dspy.OutputField(
        desc='Deduplicated JSON array of objects: [{"text": "question text", "target_field": "field_name", "retrieval_priority": 1}]'
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


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN_FALLBACK = 4
_RLM_TOKEN_THRESHOLD = 32_000

# Max pages to pass to the vision model in a single call
_VISION_CHUNK_SIZE = 10


def _estimate_tokens_accurate(text: str, model: str | None = None) -> int:
    """Estimate token count using LiteLLM's model-aware tokenizer.

    Falls back to the simple chars/4 heuristic if the tokenizer is
    unavailable (e.g. unsupported model, missing tiktoken data).
    """
    if model:
        try:
            from litellm import token_counter

            return token_counter(model=model, text=text)
        except Exception:
            logger.debug(
                "litellm.token_counter unavailable for model=%s, using fallback",
                model,
            )
    return len(text) // _CHARS_PER_TOKEN_FALLBACK


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
        self.merge_vision = dspy.Predict(ScoutMergeVisionExplorations)
        self.infer_questions = dspy.Predict(ScoutQuestionInference)
        self.dedup_questions = dspy.Predict(ScoutDeduplicateQuestions)

    def _configure_lm(self, lm: dspy.LM | None):
        if lm is not None:
            dspy.configure(lm=lm)

    def _get_model_name(self) -> str | None:
        """Extract the model identifier from the active LM for token counting."""
        if self._lm and hasattr(self._lm, "model"):
            return self._lm.model
        return None

    def _should_use_rlm(self, content: str, schema_str: str, instructions: str) -> bool:
        total_text = content + schema_str + instructions
        model = self._get_model_name()
        estimated = _estimate_tokens_accurate(total_text, model)
        return estimated >= _RLM_TOKEN_THRESHOLD

    # ------------------------------------------------------------------
    # Vision chunking (Fix 3)
    # ------------------------------------------------------------------

    def _explore_vision_chunked(
        self,
        images: list,
        schema_str: str,
        instructions: str,
    ) -> dict:
        """Process large PDFs in page-chunks then merge the results.

        If the PDF has <= _VISION_CHUNK_SIZE pages it is processed in a
        single call (same behaviour as before).
        """
        if len(images) <= _VISION_CHUNK_SIZE:
            _encode_images(images)
            result = self.explore_vision(
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

        # Chunk the pages
        num_chunks = math.ceil(len(images) / _VISION_CHUNK_SIZE)
        logger.info(
            "Chunking %d pages into %d vision batches of ≤%d pages",
            len(images),
            num_chunks,
            _VISION_CHUNK_SIZE,
        )

        partial_explorations: list[str] = []
        partial_extractions: list[dict] = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * _VISION_CHUNK_SIZE
            end = start + _VISION_CHUNK_SIZE
            chunk_images = images[start:end]

            _encode_images(chunk_images)
            result = self.explore_vision(
                extraction_schema=schema_str,
                extraction_instructions=instructions,
            )
            partial_explorations.append(result.exploration)
            try:
                partial_extractions.append(json.loads(result.extraction))
            except json.JSONDecodeError:
                partial_extractions.append({"raw": result.extraction})

            logger.info(
                "  Vision chunk %d/%d complete (%d pages)",
                chunk_idx + 1,
                num_chunks,
                len(chunk_images),
            )

        # Merge / reduce step
        self._configure_lm(self._vision_lm or self._lm)
        merge_result = self.merge_vision(
            partial_explorations=json.dumps(partial_explorations, indent=2),
            partial_extractions=json.dumps(partial_extractions, indent=2),
            extraction_schema=schema_str,
            extraction_instructions=instructions,
        )

        try:
            extraction = json.loads(merge_result.extraction)
        except json.JSONDecodeError:
            extraction = {"raw": merge_result.extraction}

        return {
            "exploration": merge_result.exploration,
            "extraction": extraction,
        }

    # ------------------------------------------------------------------
    # Main exploration entry point
    # ------------------------------------------------------------------

    def explore_document(
        self,
        content: str,
        schema: dict,
        instructions: str,
        images: list | None = None,
    ) -> dict:
        schema_str = json.dumps(schema, indent=2)

        if images:
            prev_lm = self._lm
            self._configure_lm(self._vision_lm)

            try:
                return self._explore_vision_chunked(images, schema_str, instructions)
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

    # ------------------------------------------------------------------
    # Map-Reduce question inference (Fix 2)
    # ------------------------------------------------------------------

    def infer_questions_from_explorations(
        self,
        explorations: list[str],
        schema: dict,
        instructions: str,
        batch_size: int = 5,
    ) -> list[dict]:
        """Infer retrieval questions using a Map-Reduce strategy.

        Instead of dumping *all* explorations into a single prompt
        (which can overflow context windows for large category
        bootstraps), the explorations are batched into groups of
        ``batch_size``.  Each batch produces intermediate questions
        which are then deduplicated in a final reduce step.
        """
        self._configure_lm(self._lm)
        schema_str = json.dumps(schema, indent=2)

        # --- Single-batch fast path (original behaviour) ---------------
        if len(explorations) <= batch_size:
            result = self.infer_questions(
                explorations=json.dumps(explorations, indent=2),
                extraction_schema=schema_str,
                extraction_instructions=instructions,
            )
            try:
                return json.loads(result.questions_json)
            except json.JSONDecodeError:
                return []

        # --- Map phase: batch explorations -----------------------------
        num_batches = math.ceil(len(explorations) / batch_size)
        logger.info(
            "Map-Reduce question inference: %d explorations → %d batches of ≤%d",
            len(explorations),
            num_batches,
            batch_size,
        )

        all_candidates: list[dict] = []
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch = explorations[start:end]

            result = self.infer_questions(
                explorations=json.dumps(batch, indent=2),
                extraction_schema=schema_str,
                extraction_instructions=instructions,
            )
            try:
                batch_questions = json.loads(result.questions_json)
                all_candidates.extend(batch_questions)
            except json.JSONDecodeError:
                logger.warning("Batch %d/%d returned invalid JSON", batch_idx + 1, num_batches)

        if not all_candidates:
            return []

        # --- Reduce phase: deduplicate ---------------------------------
        logger.info(
            "Reduce phase: deduplicating %d candidate questions", len(all_candidates)
        )
        dedup_result = self.dedup_questions(
            candidate_questions=json.dumps(all_candidates, indent=2),
            extraction_schema=schema_str,
        )
        try:
            return json.loads(dedup_result.questions_json)
        except json.JSONDecodeError:
            # Fall back to raw candidates if dedup fails
            return all_candidates
