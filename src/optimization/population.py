import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from src.storage import paths


class PromptCandidate(BaseModel):
    candidate_id: str
    generation: int
    created_at: str
    instructions: str
    few_shot_ids: list[str] = []
    fitness_scores: dict = {}
    parent_ids: list[str] = []
    mutation_log: str = ""


def _next_candidate_id(category: str) -> str:
    pop_dir = paths.population_dir(category)
    if not pop_dir.exists():
        return "candidate_001"
    existing = list(pop_dir.glob("candidate_*.json"))
    idx = len(existing) + 1
    return f"candidate_{idx:03d}"


def save_candidate(category: str, candidate: PromptCandidate):
    pop_dir = paths.population_dir(category)
    pop_dir.mkdir(parents=True, exist_ok=True)
    path = pop_dir / f"{candidate.candidate_id}.json"
    path.write_text(candidate.model_dump_json(indent=2))


def load_candidate(category: str, candidate_id: str) -> PromptCandidate | None:
    path = paths.population_dir(category) / f"{candidate_id}.json"
    if not path.exists():
        return None
    return PromptCandidate(**json.loads(path.read_text()))


def list_candidates(category: str) -> list[PromptCandidate]:
    pop_dir = paths.population_dir(category)
    if not pop_dir.exists():
        return []
    candidates = []
    for p in sorted(pop_dir.glob("candidate_*.json")):
        candidates.append(PromptCandidate(**json.loads(p.read_text())))
    return candidates


def save_current_prompt(category: str, candidate: PromptCandidate):
    path = paths.current_prompt_path(category)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(candidate.model_dump_json(indent=2))


def load_current_prompt(category: str) -> PromptCandidate | None:
    path = paths.current_prompt_path(category)
    if not path.exists():
        return None
    return PromptCandidate(**json.loads(path.read_text()))


def pareto_select(candidates: list[PromptCandidate]) -> list[PromptCandidate]:
    if len(candidates) <= 1:
        return candidates

    fronts = _fast_non_dominated_sort(candidates)
    return fronts[0] if fronts else candidates


def _fast_non_dominated_sort(
    candidates: list[PromptCandidate],
) -> list[list[PromptCandidate]]:
    remaining = list(candidates)
    fronts: list[list[PromptCandidate]] = []

    while remaining and len(fronts) < 10:
        front = []
        for c in remaining:
            dominated = False
            for other in remaining:
                if other.candidate_id == c.candidate_id:
                    continue
                if _dominates(other, c):
                    dominated = True
                    break
            if not dominated:
                front.append(c)
        if not front:
            break
        fronts.append(front)
        front_ids = {c.candidate_id for c in front}
        remaining = [c for c in remaining if c.candidate_id not in front_ids]

    return fronts


def _dominates(a: PromptCandidate, b: PromptCandidate) -> bool:
    a_scores = a.fitness_scores
    b_scores = b.fitness_scores

    if not a_scores or not b_scores:
        return False

    all_keys = set(a_scores.keys()) & set(b_scores.keys())
    if not all_keys:
        return False

    at_least_one_better = False
    for key in all_keys:
        a_val = float(a_scores.get(key, 0))
        b_val = float(b_scores.get(key, 0))
        if a_val < b_val:
            return False
        if a_val > b_val:
            at_least_one_better = True

    return at_least_one_better
