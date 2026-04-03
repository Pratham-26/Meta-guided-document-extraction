from enum import Enum

from pydantic import BaseModel


class QualityTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FieldDiff(BaseModel):
    field: str
    expected: str | None = None
    actual: str | None = None
    issue: str


class JudgeEvaluation(BaseModel):
    quality_tier: QualityTier
    feedback: str
    field_diffs: list[FieldDiff]
    gold_standard_id: str
    confidence: float
