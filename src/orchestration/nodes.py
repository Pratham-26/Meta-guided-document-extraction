import json
from datetime import datetime, timezone
from pathlib import Path

from src.orchestration.state import PipelineState
from src.config.loader import load_category_config, get_gold_sampling_rate
from src.config.lm import get_lm
from src.storage.fs_store import (
    has_context,
    list_gold_standards,
    save_source_document,
)
from src.storage.paths import sampling_counter_path
from src.storage.trace_logger import log_trace
from src.schemas.trace import TraceEntry
from src.schemas.document import InputType
from src.retrieval.router import RetrievalRoute, route


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


def detect_gold(state: PipelineState) -> PipelineState:
    if state.get("is_gold_doc"):
        return state

    category = state.get("category_name", "")
    modality = state.get("input_modality", "")
    threshold = get_gold_sampling_rate(category, modality)

    if threshold <= 0:
        return {**state, "is_gold_doc": False, "gold_source": None}

    counter_path = sampling_counter_path(category, modality)
    counter_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    if counter_path.exists():
        try:
            count = json.loads(counter_path.read_text()).get("count", 0)
        except (json.JSONDecodeError, OSError):
            count = 0

    count += 1
    counter_path.write_text(json.dumps({"count": count}))

    if count >= threshold:
        counter_path.write_text(json.dumps({"count": 0}))
        return {**state, "is_gold_doc": True, "gold_source": "random_sample"}

    return {**state, "is_gold_doc": False, "gold_source": None}


def run_scout_for_gold(state: PipelineState) -> PipelineState:
    from src.agents.scout.agent import ScoutAgent
    from src.agents.scout.gold_builder import build_and_save
    from src.agents.scout.question_store import merge_questions
    from src.utils.pdf import extract_text_from_pdf, load_pdf_pages
    from src.utils.text import clean_text, truncate_to_tokens

    category = state.get("category_name", "")
    modality = state.get("input_modality", "")
    doc = state.get("document")
    schema = state.get("schema", {})
    instructions = state.get("instructions", "")

    if not doc:
        return {**state, "error": "No document provided for gold processing."}

    path = doc.source_uri
    is_pdf = modality == "pdf"

    saved = save_source_document(category, modality, path)

    images = None
    if is_pdf:
        content = extract_text_from_pdf(path)
        content = clean_text(content)
        content = truncate_to_tokens(content)
        is_placeholder = content.startswith("[PDF with")

        if is_placeholder:
            images = load_pdf_pages(path)

        lm = get_lm("scout", input_type="text")
        vision_lm = get_lm("scout", input_type="vision")
        scout = ScoutAgent(lm=lm, vision_lm=vision_lm)
    else:
        content = path.read_text(encoding="utf-8")
        content = clean_text(content)
        content = truncate_to_tokens(content)

        lm = get_lm("scout", input_type="text")
        scout = ScoutAgent(lm=lm)

    result = scout.explore_document(
        content=content,
        schema=schema,
        instructions=instructions,
        images=images,
    )

    existing_gs = list_gold_standards(category, modality)
    next_num = 1
    if existing_gs:
        nums = []
        for gs in existing_gs:
            if gs.id.startswith("gs_"):
                try:
                    nums.append(int(gs.id.split("_")[1]))
                except (ValueError, IndexError):
                    pass
        if nums:
            next_num = max(nums) + 1

    gs_id = f"gs_{next_num:03d}"
    build_and_save(
        category=category,
        modality=modality,
        gs_id=gs_id,
        source_document_uri=saved,
        extraction=result["extraction"],
    )

    questions = scout.infer_questions_from_explorations(
        explorations=[result["exploration"]],
        schema=schema,
        instructions=instructions,
    )

    if questions:
        merge_questions(category, modality, questions)

    return state


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
        gold_standard_id=gs.id,
    )

    traces = state.get("trace_entries", []) + [trace]
    return {**state, "judge_evaluation": evaluation, "trace_entries": traces}


def log_traces(state: PipelineState) -> PipelineState:
    traces = state.get("trace_entries", [])
    for trace in traces:
        log_trace(trace)
    return state
