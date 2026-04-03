import dspy

from src.config.loader import load_model_config, ModelConfig

_active_lms: dict[str, dspy.LM] = {}


def get_lm(agent_role: str, config: ModelConfig | None = None) -> dspy.LM:
    if agent_role in _active_lms:
        return _active_lms[agent_role]

    if config is None:
        config = load_model_config()

    role_config = config.agent_roles[agent_role]
    lm = dspy.LM(
        model=role_config.model,
        temperature=role_config.temperature,
        max_tokens=role_config.max_tokens,
    )
    _active_lms[agent_role] = lm
    return lm


def clear_lm_cache():
    _active_lms.clear()
