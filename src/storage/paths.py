from pathlib import Path

from src.config.settings import settings


def category_dir(name: str) -> Path:
    return settings.categories_dir / name


def gold_standards_dir(name: str) -> Path:
    return category_dir(name) / "gold_standards"


def gold_standard_path(name: str, gs_id: str) -> Path:
    return gold_standards_dir(name) / f"{gs_id}.json"


def sources_dir(name: str) -> Path:
    return gold_standards_dir(name) / "sources"


def questions_path(name: str) -> Path:
    return category_dir(name) / "questions" / "questions.json"


def colpali_index_dir(name: str) -> Path:
    return category_dir(name) / "indexes" / "colpali"


def colbert_index_dir(name: str) -> Path:
    return category_dir(name) / "indexes" / "colbert"


def prompts_dir(name: str) -> Path:
    return category_dir(name) / "prompts"


def current_prompt_path(name: str) -> Path:
    return prompts_dir(name) / "current.json"


def population_dir(name: str) -> Path:
    return prompts_dir(name) / "population"


def trace_traces_dir(name: str) -> Path:
    return settings.traces_dir / name


def extraction_traces_dir(name: str) -> Path:
    return trace_traces_dir(name) / "extraction_traces"


def judge_traces_dir(name: str) -> Path:
    return trace_traces_dir(name) / "judge_traces"


def optimization_traces_dir(name: str) -> Path:
    return trace_traces_dir(name) / "optimization_traces"


def ensure_category_dirs(name: str):
    for d in [
        gold_standards_dir(name),
        sources_dir(name),
        category_dir(name) / "questions",
        colpali_index_dir(name),
        colbert_index_dir(name),
        prompts_dir(name),
        population_dir(name),
    ]:
        d.mkdir(parents=True, exist_ok=True)


def ensure_trace_dirs(name: str):
    for d in [
        extraction_traces_dir(name),
        judge_traces_dir(name),
        optimization_traces_dir(name),
    ]:
        d.mkdir(parents=True, exist_ok=True)
