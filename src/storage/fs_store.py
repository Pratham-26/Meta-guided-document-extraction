import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from src.schemas.gold_standard import ApprovalStatus, GoldStandard
from src.schemas.question import QuestionSet
from src.storage import paths


def _atomic_write(path: Path, data: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_gold_standard(category: str, modality: str, gs: GoldStandard):
    paths.ensure_category_dirs(category)
    path = paths.gold_standard_path(category, modality, gs.id)
    _atomic_write(path, gs.model_dump_json(indent=2))


def load_gold_standard(category: str, modality: str, gs_id: str) -> GoldStandard:
    path = paths.gold_standard_path(category, modality, gs_id)
    with open(path) as f:
        return GoldStandard(**json.load(f))


def list_gold_standards(category: str, modality: str) -> list[GoldStandard]:
    """Return *all* gold standards regardless of approval status."""
    gs_dir = paths.gold_standards_dir(category, modality)
    if not gs_dir.exists():
        return []
    results = []
    for p in sorted(gs_dir.glob("gs_*.json")):
        with open(p) as f:
            results.append(GoldStandard(**json.load(f)))
    return results


def list_approved_gold_standards(
    category: str, modality: str
) -> list[GoldStandard]:
    """Return only gold standards that have been explicitly approved.

    GEPA optimization and the Judge agent should call this instead of
    ``list_gold_standards`` to avoid training against unverified Scout
    extractions.
    """
    return [
        gs
        for gs in list_gold_standards(category, modality)
        if gs.approval_status == ApprovalStatus.APPROVED
    ]


def approve_gold_standard(
    category: str, modality: str, gs_id: str, *, approved_by: str = "human"
) -> GoldStandard:
    """Promote a pending-review gold standard to approved status."""
    gs = load_gold_standard(category, modality, gs_id)
    gs.approval_status = ApprovalStatus.APPROVED
    gs.approved_by = approved_by
    save_gold_standard(category, modality, gs)
    return gs


def reject_gold_standard(category: str, modality: str, gs_id: str) -> GoldStandard:
    """Mark a gold standard as rejected so it is excluded everywhere."""
    gs = load_gold_standard(category, modality, gs_id)
    gs.approval_status = ApprovalStatus.REJECTED
    save_gold_standard(category, modality, gs)
    return gs


def delete_gold_standard(category: str, modality: str, gs_id: str) -> bool:
    path = paths.gold_standard_path(category, modality, gs_id)
    if path.exists():
        path.unlink()
        return True
    return False


def save_source_document(category: str, modality: str, source_path: Path):
    dest_dir = paths.sources_dir(category, modality)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source_path.name
    if not dest.exists():
        shutil.copy2(source_path, dest)
    return dest


def save_question_set(category: str, modality: str, qs: QuestionSet):
    paths.ensure_category_dirs(category)
    path = paths.questions_path(category, modality)
    _atomic_write(path, qs.model_dump_json(indent=2))


def load_question_set(category: str, modality: str) -> QuestionSet | None:
    path = paths.questions_path(category, modality)
    if not path.exists():
        return None
    with open(path) as f:
        return QuestionSet(**json.load(f))


def has_context(category: str, modality: str) -> bool:
    q_path = paths.questions_path(category, modality)
    gs_dir = paths.gold_standards_dir(category, modality)
    return q_path.exists() and gs_dir.exists() and any(gs_dir.glob("gs_*.json"))
