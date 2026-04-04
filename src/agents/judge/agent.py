import dspy
import json

from src.schemas.evaluation import FieldDiff, JudgeEvaluation, QualityTier


class JudgeCompare(dspy.Signature):
    """Compare an extraction against a Gold Standard and evaluate quality.
    Identify specific field-level differences and assign a quality tier."""

    extraction: str = dspy.InputField(desc="The extraction result to evaluate (JSON)")
    gold_standard: str = dspy.InputField(
        desc="The approved Gold Standard extraction (JSON)"
    )
    extraction_schema: str = dspy.InputField(
        desc="JSON schema defining expected fields"
    )
    quality_tier: str = dspy.OutputField(desc="One of: low, medium, high")
    feedback: str = dspy.OutputField(
        desc="Textual explanation of divergences between extraction and gold standard"
    )
    field_diffs: str = dspy.OutputField(
        desc='JSON array of field diffs: [{"field": "name", "expected": "value", "actual": "value", "issue": "description"}]'
    )
    confidence: str = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")


class JudgeAgent:
    def __init__(self, lm: dspy.LM | None = None):
        if lm:
            dspy.configure(lm=lm)
        self.compare = dspy.Predict(JudgeCompare)

    def evaluate(
        self,
        extraction: dict,
        gold_standard: dict,
        schema: dict,
        gold_standard_id: str,
    ) -> JudgeEvaluation:
        result = self.compare(
            extraction=json.dumps(extraction, indent=2),
            gold_standard=json.dumps(gold_standard, indent=2),
            extraction_schema=json.dumps(schema, indent=2),
        )

        tier_str = result.quality_tier.strip().lower()
        try:
            quality_tier = QualityTier(tier_str)
        except ValueError:
            quality_tier = QualityTier.MEDIUM

        try:
            diffs_data = json.loads(result.field_diffs)
            field_diffs = [FieldDiff(**d) for d in diffs_data]
        except (json.JSONDecodeError, Exception):
            field_diffs = []

        try:
            confidence = float(result.confidence.strip())
        except ValueError:
            confidence = 0.5

        return JudgeEvaluation(
            quality_tier=quality_tier,
            feedback=result.feedback,
            field_diffs=field_diffs,
            gold_standard_id=gold_standard_id,
            confidence=max(0.0, min(1.0, confidence)),
        )
