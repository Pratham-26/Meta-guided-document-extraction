from src.orchestration.state import PipelineState
from src.orchestration import nodes


def run_pipeline(state: PipelineState) -> PipelineState:
    state = nodes.check_context(state)
    if state.get("error"):
        return state

    state = nodes.resolve_config(state)
    state = nodes.detect_gold(state)

    if state.get("is_gold_doc"):
        state = nodes.run_scout_for_gold(state)

    state = nodes.load_questions(state)
    if state.get("error"):
        return state

    state = nodes.route_input(state)
    if state.get("error"):
        return state

    state = nodes.retrieve(state)
    if state.get("error"):
        return state

    state = nodes.extract(state)

    if state.get("is_gold_doc"):
        state = nodes.judge(state)

    state = nodes.log_traces(state)
    state = nodes.cleanup_index(state)

    return state
