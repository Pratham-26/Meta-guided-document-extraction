from datetime import datetime, timezone
from pathlib import Path

from src.schemas.gold_standard import ApprovalStatus, GoldStandard
from src.storage.fs_store import save_gold_standard


def build_and_save(
    category: str,
    modality: str,
    gs_id: str,
    source_document_uri: Path,
    extraction: dict,
    approved_by: str = "scout",
    approval_status: ApprovalStatus = ApprovalStatus.PENDING_REVIEW,
) -> GoldStandard:
    """Create and persist a new Gold Standard.

    By default Scout-generated extractions are saved as
    ``PENDING_REVIEW`` so they must be explicitly approved via the
    HITL review UI before GEPA or the Judge trusts them.
    """
    gs = GoldStandard(
        id=gs_id,
        category=category,
        input_modality=modality,
        source_document_uri=source_document_uri,
        extraction=extraction,
        approved_by=approved_by,
        created_at=datetime.now(timezone.utc),
        supersedes=None,
        approval_status=approval_status,
    )
    save_gold_standard(category, modality, gs)
    return gs
