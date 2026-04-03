from pathlib import Path

from src.config.settings import settings


_VALID_MODALITIES = {"pdf", "text"}


def _validate_modality(modality: str):
    if modality not in _VALID_MODALITIES:
        raise ValueError(
            f"Invalid modality '{modality}'. Must be one of: {_VALID_MODALITIES}"
        )


def category_dir(name: str) -> Path:
    return settings.categories_dir / name


def modality_dir(name: str, modality: str) -> Path:
    _validate_modality(modality)
    return category_dir(name) / modality


def gold_standards_dir(name: str, modality: str) -> Path:
    return modality_dir(name, modality) / "gold_standards"


def gold_standard_path(name: str, modality: str, gs_id: str) -> Path:
    return gold_standards_dir(name, modality) / f"{gs_id}.json"


def sources_dir(name: str, modality: str) -> Path:
    return gold_standards_dir(name, modality) / "sources"


def questions_path(name: str, modality: str) -> Path:
    return modality_dir(name, modality) / "questions" / "questions.json"


def colpali_index_dir(name: str) -> Path:
    return category_dir(name) / "pdf" / "indexes" / "colpali"


def colbert_index_dir(name: str) -> Path:
    return category_dir(name) / "text" / "indexes" / "colbert"


def prompts_dir(name: str, modality: str) -> Path:
    return modality_dir(name, modality) / "prompts"


def current_prompt_path(name: str, modality: str) -> Path:
    return prompts_dir(name, modality) / "current.json"


def population_dir(name: str, modality: str) -> Path:
    return prompts_dir(name, modality) / "population"


def trace_traces_dir(name: str, modality: str) -> Path:
    _validate_modality(modality)
    return settings.traces_dir / name / modality


def extraction_traces_dir(name: str, modality: str) -> Path:
    return trace_traces_dir(name, modality) / "extraction_traces"


def judge_traces_dir(name: str, modality: str) -> Path:
    return trace_traces_dir(name, modality) / "judge_traces"


def optimization_traces_dir(name: str, modality: str) -> Path:
    return trace_traces_dir(name, modality) / "optimization_traces"


def ensure_category_dirs(name: str):
    for modality in _VALID_MODALITIES:
        for d in [
            gold_standards_dir(name, modality),
            sources_dir(name, modality),
            modality_dir(name, modality) / "questions",
            prompts_dir(name, modality),
            population_dir(name, modality),
        ]:
            d.mkdir(parents=True, exist_ok=True)
    colpali_index_dir(name).mkdir(parents=True, exist_ok=True)
    colbert_index_dir(name).mkdir(parents=True, exist_ok=True)


def ensure_trace_dirs(name: str):
    for modality in _VALID_MODALITIES:
        for d in [
            extraction_traces_dir(name, modality),
            judge_traces_dir(name, modality),
            optimization_traces_dir(name, modality),
        ]:
            d.mkdir(parents=True, exist_ok=True)
