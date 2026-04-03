from __future__ import annotations

from typing import TypedDict

from src.schemas.document import DocumentInput
from src.schemas.evaluation import JudgeEvaluation
from src.schemas.trace import TraceEntry
from src.retrieval.router import RetrievalRoute


class PipelineState(TypedDict, total=False):
    document: DocumentInput
    category_name: str
    input_modality: str
    schema: dict
    instructions: str
    retrieval_route: RetrievalRoute | None
    questions: list[str]
    retrieved_context: str | None
    retrieved_images: list | None
    extraction: dict | None
    judge_evaluation: JudgeEvaluation | None
    trace_entries: list[TraceEntry]
    error: str | None
    human_corrections: dict | None
