from langgraph.graph import StateGraph, START, END

from src.orchestration.state import PipelineState
from src.orchestration import nodes


def _has_error(state: PipelineState) -> str:
    if state.get("error"):
        return "halt"
    return "continue"


def _route_retrieval(state: PipelineState) -> str:
    r = state.get("retrieval_route")
    if r:
        return r.value
    return "colbert"


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("check_context", nodes.check_context)
    graph.add_node("resolve_config", nodes.resolve_config)
    graph.add_node("load_questions", nodes.load_questions)
    graph.add_node("route_input", nodes.route_input)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("extract", nodes.extract)
    graph.add_node("judge", nodes.judge)
    graph.add_node("log_traces", nodes.log_traces)

    graph.add_edge(START, "check_context")
    graph.add_conditional_edges(
        "check_context",
        _has_error,
        {
            "halt": END,
            "continue": "resolve_config",
        },
    )
    graph.add_edge("resolve_config", "load_questions")
    graph.add_edge("load_questions", "route_input")
    graph.add_edge("route_input", "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "judge")
    graph.add_edge("judge", "log_traces")
    graph.add_edge("log_traces", END)

    return graph


def compile_graph():
    return build_graph().compile()
