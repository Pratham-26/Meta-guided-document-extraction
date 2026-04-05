from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class ApprovalStatus(str, Enum):
    """Tracks whether a Gold Standard has been human-verified.

    Scout-generated extractions start as ``PENDING_REVIEW`` ("Silver
    Standards") and must be explicitly promoted to ``APPROVED`` before
    they are trusted by GEPA optimization or the Judge agent.
    """

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class GoldStandard(BaseModel):
    id: str
    category: str
    input_modality: str
    source_document_uri: Path
    extraction: dict
    approved_by: str
    created_at: datetime
    supersedes: str | None = None
    approval_status: ApprovalStatus = ApprovalStatus.PENDING_REVIEW
