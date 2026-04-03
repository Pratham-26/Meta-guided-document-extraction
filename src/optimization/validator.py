from src.agents.judge.agent import JudgeAgent
from src.agents.extractor.agent import ExtractorAgent
from src.storage.fs_store import list_gold_standards
from src.schemas.evaluation import QualityTier
from src.config.lm import get_lm


def validate_candidate(
    category: str,
    modality: str,
    instructions: str,
    schema: dict,
    sample_size: int = 10,
) -> dict:
    gold_standards = list_gold_standards(category, modality)
    if not gold_standards:
        return {"error": "No Gold Standards available for validation."}

    samples = gold_standards[:sample_size]

    extractor_lm = get_lm("extractor")
    judge_lm = get_lm("judge")
    extractor = ExtractorAgent(lm=extractor_lm)
    judge_agent = JudgeAgent(lm=judge_lm)

    results = {"total": len(samples), "low": 0, "medium": 0, "high": 0, "details": []}

    for gs in samples:
        context = json.dumps(gs.extraction, indent=2)
        extraction = extractor.run(
            context=context,
            schema=schema,
            instructions=instructions,
        )

        evaluation = judge_agent.evaluate(
            extraction=extraction,
            gold_standard=gs.extraction,
            schema=schema,
            gold_standard_id=gs.id,
        )

        tier = evaluation.quality_tier.value
        results[tier] += 1
        results["details"].append(
            {
                "gold_standard_id": gs.id,
                "quality_tier": tier,
                "confidence": evaluation.confidence,
            }
        )

    results["accuracy"] = (
        results["high"] / results["total"] if results["total"] > 0 else 0.0
    )
    return results


import json
