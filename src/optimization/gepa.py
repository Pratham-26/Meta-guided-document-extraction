from datetime import datetime, timezone

from src.config.lm import get_lm
from src.config.loader import load_category_config
from src.storage.fs_store import list_approved_gold_standards
from src.storage.trace_logger import log_trace
from src.schemas.trace import TraceEntry
from src.schemas.evaluation import QualityTier
from src.optimization.reflector import Reflector, PromptMutator
from src.optimization.population import (
    PromptCandidate,
    _next_candidate_id,
    save_candidate,
    list_candidates,
    pareto_select,
    save_current_prompt,
    load_current_prompt,
)
from src.optimization.validator import validate_candidate


def run_gepa_cycle(
    category: str,
    modality: str,
    generations: int | None = None,
    population_size: int | None = None,
) -> dict:
    config = load_category_config(category)
    generations = generations or config.optimization.gepa_generations
    population_size = population_size or config.optimization.gepa_population_size

    gold_standards = list_approved_gold_standards(category, modality)
    if not gold_standards:
        return {"error": "No approved Gold Standards available for optimization."}

    current = load_current_prompt(category, modality)
    if not current:
        current = PromptCandidate(
            candidate_id="candidate_000",
            generation=0,
            created_at=datetime.now(timezone.utc).isoformat(),
            instructions=config.extraction_instructions,
        )
        save_candidate(category, modality, current)

    reflector_lm = get_lm("reflector")
    reflector = Reflector(lm=reflector_lm)
    mutator = PromptMutator(lm=reflector_lm)

    trace_entries = []

    for gen in range(1, generations + 1):
        low_medium_gs = _select_low_medium_samples(category, modality, gold_standards)
        if not low_medium_gs:
            continue

        for gs_sample in low_medium_gs[:population_size]:
            analysis = reflector.analyze(
                extraction=gs_sample.extraction,
                gold_standard=gs_sample.extraction,
                schema=config.expected_schema,
                current_instructions=current.instructions,
                field_feedback="Optimization cycle analysis",
            )

            mutation = mutator.mutate(
                current_instructions=current.instructions,
                diagnosis=analysis["diagnosis"],
                suggested_fixes=analysis["suggested_fixes"],
            )

            new_id = _next_candidate_id(category, modality)
            candidate = PromptCandidate(
                candidate_id=new_id,
                generation=gen,
                created_at=datetime.now(timezone.utc).isoformat(),
                instructions=mutation["revised_instructions"],
                parent_ids=[current.candidate_id],
                mutation_log=mutation["mutation_rationale"],
            )
            save_candidate(category, modality, candidate)

        all_candidates = list_candidates(category, modality)
        for c in all_candidates:
            if c.fitness_scores:
                continue
            validation = validate_candidate(
                category,
                modality,
                c.instructions,
                config.expected_schema,
                sample_size=config.optimization.validation_sample_size,
            )
            c.fitness_scores = {
                "overall_accuracy": validation.get("accuracy", 0.0),
                "high_count": validation.get("high", 0),
                "total": validation.get("total", 1),
            }
            save_candidate(category, modality, c)

        all_candidates = list_candidates(category, modality)
        front = pareto_select(all_candidates)
        if front:
            best = max(front, key=lambda c: c.fitness_scores.get("overall_accuracy", 0))
            current = best
            save_current_prompt(category, modality, best)

        trace = TraceEntry(
            timestamp=datetime.now(timezone.utc),
            agent_role="reflector",
            phase="optimization",
            category=category,
            input_modality=modality,
            prompt=current.instructions,
            response=f"Generation {gen} complete. Best candidate: {current.candidate_id}",
            model="reflector",
            provider="litellm",
            token_usage={},
        )
        log_trace(trace)

    return {
        "generations_run": generations,
        "final_candidate": current.candidate_id,
        "fitness_scores": current.fitness_scores,
    }


def _select_low_medium_samples(
    category: str, modality: str, gold_standards: list
) -> list:
    from src.storage.trace_logger import read_traces
    from src.schemas.evaluation import QualityTier

    low_medium_gs_ids = set()
    try:
        judge_traces = read_traces(category, modality, phase="evaluation")
        for trace in judge_traces:
            if trace.quality_tier in (QualityTier.LOW.value, QualityTier.MEDIUM.value):
                low_medium_gs_ids.add(trace.gold_standard_id)
    except Exception:
        pass

    if low_medium_gs_ids:
        return [gs for gs in gold_standards if gs.id in low_medium_gs_ids]

    return gold_standards
