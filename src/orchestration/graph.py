from langgraph.graph import StateGraph, START, END

from src.orchestration.state import PipelineState
from src.orchestration import nodes


def _has_error(state: PipelineState) -> str:
    if state.get("error"):
        return "halt"
    return "continue"


def _is_gold(state: PipelineState) -> str:
    if state.get("is_gold_doc"):
        return "gold"
    return "regular"


def _should_judge(state: PipelineState) -> str:
    if state.get("is_gold_doc"):
        return "judge"
    return "skip"


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("check_context", nodes.check_context)
    graph.add_node("resolve_config", nodes.resolve_config)
    graph.add_node("detect_gold", nodes.detect_gold)
    graph.add_node("run_scout_for_gold", nodes.run_scout_for_gold)
    graph.add_node("load_questions", nodes.load_questions)
    graph.add_node("route_input", nodes.route_input)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("extract", nodes.extract)
    graph.add_node("judge", nodes.judge)
    graph.add_node("log_traces", nodes.log_traces)
    graph.add_node("cleanup_index", nodes.cleanup_index)

    graph.add_edge(START, "check_context")
    graph.add_conditional_edges(
        "check_context",
        _has_error,
        {
            "halt": END,
            "continue": "resolve_config",
        },
    )
    graph.add_edge("resolve_config", "detect_gold")
    graph.add_conditional_edges(
        "detect_gold",
        _is_gold,
        {
            "gold": "run_scout_for_gold",
            "regular": "load_questions",
        },
    )
    graph.add_edge("run_scout_for_gold", "load_questions")
    graph.add_edge("load_questions", "route_input")
    graph.add_edge("route_input", "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_conditional_edges(
        "extract",
        _should_judge,
        {
            "judge": "judge",
            "skip": "log_traces",
        },
    )
    graph.add_edge("judge", "log_traces")
    graph.add_edge("log_traces", "cleanup_index")
    graph.add_edge("cleanup_index", END)

    return graph


def compile_graph():
    return build_graph().compile()
