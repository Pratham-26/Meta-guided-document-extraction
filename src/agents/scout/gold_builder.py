from datetime import datetime, timezone
from pathlib import Path

from src.schemas.gold_standard import GoldStandard
from src.storage.fs_store import save_gold_standard


def build_and_save(
    category: str,
    gs_id: str,
    source_document_uri: Path,
    extraction: dict,
    approved_by: str = "scout",
) -> GoldStandard:
    gs = GoldStandard(
        id=gs_id,
        category=category,
        source_document_uri=source_document_uri,
        extraction=extraction,
        approved_by=approved_by,
        created_at=datetime.now(timezone.utc),
        supersedes=None,
    )
    save_gold_standard(category, gs)
    return gs
