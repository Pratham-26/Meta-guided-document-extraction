from datetime import datetime, timezone

from src.orchestration.state import PipelineState
from src.config.loader import load_category_config
from src.config.lm import get_lm
from src.storage.fs_store import has_context, list_gold_standards
from src.storage.trace_logger import log_trace
from src.schemas.trace import TraceEntry
from src.schemas.document import InputType
from src.retrieval.router import RetrievalRoute, route


def check_context(state: PipelineState) -> PipelineState:
    category = state.get("category_name", "")
    if not has_context(category):
        return {
            **state,
            "error": f"No Scout context for category '{category}'. Run bootstrapping first.",
        }
    return state


def resolve_config(state: PipelineState) -> PipelineState:
    category = state.get("category_name", "")
    config = load_category_config(category)
    return {
        **state,
        "schema": config.expected_schema,
        "instructions": config.extraction_instructions,
    }


def load_questions(state: PipelineState) -> PipelineState:
    from src.agents.scout.question_store import get_questions

    category = state.get("category_name", "")
    questions = get_questions(category)
    if not questions:
        return {**state, "error": f"No questions found for category '{category}'."}
    return {**state, "questions": questions}


def route_input(state: PipelineState) -> PipelineState:
    doc = state.get("document")
    if not doc:
        return {**state, "error": "No document provided."}
    retrieval_route = route(doc.input_type, doc.source_uri)
    return {**state, "retrieval_route": retrieval_route}


def retrieve(state: PipelineState) -> PipelineState:
    category = state.get("category_name", "")
    questions = state.get("questions", [])
    doc = state.get("document")
    retrieval_route = state.get("retrieval_route")

    try:
        if retrieval_route == RetrievalRoute.COLPALI:
            from src.retrieval.colpali.retriever import get_retrieved_pages

            config = load_category_config(category)
            pages = get_retrieved_pages(
                category, questions, config.retrieval.colpali_top_k
            )
            context_parts = [f"Page {p['page_number']}" for p in pages]
            context = f"Retrieved pages (visual):\n" + "\n".join(context_parts)
        else:
            from src.retrieval.colbert.retriever import get_retrieved_chunks

            config = load_category_config(category)
            chunks = get_retrieved_chunks(
                category, questions, config.retrieval.colbert_top_k
            )
            context_parts = [f"[Chunk {c['rank']}] {c['content']}" for c in chunks]
            context = "Retrieved text chunks:\n" + "\n".join(context_parts)

        return {**state, "retrieved_context": context}
    except Exception as e:
        return {**state, "error": f"Retrieval failed: {e}"}


def extract(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    from src.agents.extractor.agent import ExtractorAgent
    from src.agents.extractor.few_shot import select_examples

    category = state.get("category_name", "")
    lm = get_lm("extractor")
    agent = ExtractorAgent(lm=lm)

    examples = select_examples(category)
    extraction = agent.run(
        context=state.get("retrieved_context", ""),
        schema=state.get("schema", {}),
        instructions=state.get("instructions", ""),
        few_shot_examples=examples,
    )

    trace = TraceEntry(
        timestamp=datetime.now(timezone.utc),
        agent_role="extractor",
        phase="extraction",
        category=category,
        prompt=state.get("retrieved_context", ""),
        response=str(extraction),
        model="extractor",
        provider="litellm",
        token_usage={},
        document_id=str(
            state.get("document", {}).source_uri if state.get("document") else "unknown"
        ),
    )

    traces = state.get("trace_entries", []) + [trace]
    return {**state, "extraction": extraction, "trace_entries": traces}


def judge(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    category = state.get("category_name", "")
    gold_standards = list_gold_standards(category)
    if not gold_standards:
        return state

    from src.agents.judge.agent import JudgeAgent

    lm = get_lm("judge")
    agent = JudgeAgent(lm=lm)
    gs = gold_standards[0]

    evaluation = agent.evaluate(
        extraction=state.get("extraction", {}),
        gold_standard=gs.extraction,
        schema=state.get("schema", {}),
        gold_standard_id=gs.id,
    )

    trace = TraceEntry(
        timestamp=datetime.now(timezone.utc),
        agent_role="judge",
        phase="evaluation",
        category=category,
        prompt=str(state.get("extraction", {})),
        response=evaluation.model_dump_json(),
        model="judge",
        provider="litellm",
        token_usage={},
        quality_tier=evaluation.quality_tier.value,
    )

    traces = state.get("trace_entries", []) + [trace]
    return {**state, "judge_evaluation": evaluation, "trace_entries": traces}


def log_traces(state: PipelineState) -> PipelineState:
    traces = state.get("trace_entries", [])
    for trace in traces:
        log_trace(trace)
    return state
