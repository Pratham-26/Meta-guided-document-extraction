from src.orchestration.state import PipelineState


def present_for_review(state: PipelineState) -> dict:
    summary = {
        "category": state.get("category_name"),
        "questions": state.get("questions", []),
        "extraction": state.get("extraction"),
    }
    return summary


def apply_human_corrections(state: PipelineState, corrections: dict) -> PipelineState:
    updated = {**state}
    if "extraction" in corrections:
        updated["extraction"] = corrections["extraction"]
    if "questions" in corrections:
        updated["questions"] = corrections["questions"]
    return updated
