import json
from datetime import datetime, timezone
from pathlib import Path

from src.schemas.trace import TraceEntry
from src.storage import paths


_PHASE_TO_DIR = {
    "extraction": paths.extraction_traces_dir,
    "evaluation": paths.judge_traces_dir,
    "optimization": paths.optimization_traces_dir,
    "bootstrap": paths.extraction_traces_dir,
}


def _trace_file(category: str, phase: str) -> Path:
    phase_dir_fn = _PHASE_TO_DIR.get(phase, paths.extraction_traces_dir)
    phase_dir = phase_dir_fn(category)
    phase_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return phase_dir / f"trace_{today}.jsonl"


def log_trace(entry: TraceEntry):
    path = _trace_file(entry.category, entry.phase)
    with open(path, "a") as f:
        f.write(entry.model_dump_json() + "\n")


def log_traces(entries: list[TraceEntry]):
    if not entries:
        return
    grouped: dict[Path, list[TraceEntry]] = {}
    for e in entries:
        p = _trace_file(e.category, e.phase)
        grouped.setdefault(p, []).append(e)
    for path, batch in grouped.items():
        with open(path, "a") as f:
            for e in batch:
                f.write(e.model_dump_json() + "\n")


def read_traces(category: str, phase: str, date: str | None = None) -> list[TraceEntry]:
    if date:
        phase_dir_fn = _PHASE_TO_DIR.get(phase, paths.extraction_traces_dir)
        path = phase_dir_fn(category) / f"trace_{date}.jsonl"
        if not path.exists():
            return []
        with open(path) as f:
            return [TraceEntry(**json.loads(line)) for line in f if line.strip()]

    phase_dir_fn = _PHASE_TO_DIR.get(phase, paths.extraction_traces_dir)
    phase_dir = phase_dir_fn(category)
    if not phase_dir.exists():
        return []
    results = []
    for p in sorted(phase_dir.glob("trace_*.jsonl")):
        with open(p) as f:
            for line in f:
                if line.strip():
                    results.append(TraceEntry(**json.loads(line)))
    return results
