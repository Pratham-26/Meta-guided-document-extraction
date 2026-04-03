from datetime import datetime, timezone
import base64
import io

from src.orchestration.state import PipelineState
from src.config.loader import load_category_config
from src.config.lm import get_lm
from src.storage.fs_store import has_context, list_gold_standards
from src.storage.trace_logger import log_trace
from src.schemas.trace import TraceEntry
from src.schemas.document import InputType
from src.retrieval.router import RetrievalRoute, route


def _encode_pil_image(image) -> str:
    import PIL.Image

    buf = io.BytesIO()
    img = image if isinstance(image, PIL.Image.Image) else PIL.Image.open(image)
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_model_provider(model_string: str) -> tuple[str, str]:
    if "/" in model_string:
        provider, model = model_string.split("/", 1)
        return model, provider
    return model_string, "unknown"


def check_context(state: PipelineState) -> PipelineState:
    category = state.get("category_name", "")
    modality = state.get("input_modality", "")
    if not has_context(category, modality):
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
    modality = state.get("input_modality", "")
    questions = get_questions(category, modality)
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
            images = [p["image"] for p in pages if "image" in p]
            return {
                **state,
                "retrieved_context": context,
                "retrieved_images": images if images else None,
            }
        else:
            from src.retrieval.colbert.retriever import get_retrieved_chunks

            config = load_category_config(category)
            chunks = get_retrieved_chunks(
                category, questions, config.retrieval.colbert_top_k
            )
            context_parts = [f"[Chunk {c['rank']}] {c['content']}" for c in chunks]
            context = "Retrieved text chunks:\n" + "\n".join(context_parts)
            return {**state, "retrieved_context": context, "retrieved_images": None}

    except Exception as e:
        return {**state, "error": f"Retrieval failed: {e}"}


def extract(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    from src.agents.extractor.agent import ExtractorAgent
    from src.agents.extractor.few_shot import select_examples

    category = state.get("category_name", "")
    modality = state.get("input_modality", "")
    retrieval_route = state.get("retrieval_route")
    images = state.get("retrieved_images")

    input_type = (
        "vision" if images and retrieval_route == RetrievalRoute.COLPALI else "text"
    )
    lm = get_lm("extractor", input_type=input_type)
    agent = ExtractorAgent(lm=lm)

    examples = select_examples(category, modality)
    extraction = agent.run(
        context=state.get("retrieved_context", ""),
        schema=state.get("schema", {}),
        instructions=state.get("instructions", ""),
        few_shot_examples=examples,
        images=images,
    )

    trace = TraceEntry(
        timestamp=datetime.now(timezone.utc),
        agent_role="extractor",
        phase="extraction",
        category=category,
        input_modality=modality,
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


def _pick_best_gold_standard(extraction: dict, gold_standards: list) -> object:
    if len(gold_standards) == 1:
        return gold_standards[0]

    extraction_keys = set(
        str(k).lower() for k in extraction.keys() if extraction.get(k) is not None
    )
    best_score = -1
    best_gs = gold_standards[0]

    for gs in gold_standards:
        gs_keys = set(
            str(k).lower()
            for k in gs.extraction.keys()
            if gs.extraction.get(k) is not None
        )
        overlap = len(extraction_keys & gs_keys)
        if overlap > best_score:
            best_score = overlap
            best_gs = gs

    return best_gs


def judge(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    category = state.get("category_name", "")
    modality = state.get("input_modality", "")
    gold_standards = list_gold_standards(category, modality)
    if not gold_standards:
        return state

    from src.agents.judge.agent import JudgeAgent

    lm = get_lm("judge")
    agent = JudgeAgent(lm=lm)
    gs = _pick_best_gold_standard(state.get("extraction", {}), gold_standards)

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
        input_modality=modality,
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
