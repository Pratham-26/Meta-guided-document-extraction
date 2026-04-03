import dspy

from src.config.loader import load_model_config, ModelConfig

_active_lms: dict[str, dspy.LM] = {}


def get_lm(
    agent_role: str,
    input_type: str = "text",
    config: ModelConfig | None = None,
) -> dspy.LM:
    cache_key = f"{agent_role}:{input_type}"
    if cache_key in _active_lms:
        return _active_lms[cache_key]

    if config is None:
        config = load_model_config()

    role_config = config.agent_roles[agent_role]
    model_name = role_config.get_model(input_type)
    lm = dspy.LM(
        model=model_name,
        temperature=role_config.temperature,
        max_tokens=role_config.max_tokens,
    )
    _active_lms[cache_key] = lm
    return lm


def clear_lm_cache():
    _active_lms.clear()
