import json
from pathlib import Path

from pydantic import BaseModel, field_validator

from src.config.settings import settings


class RetrievalConfig(BaseModel):
    default_route: str = "auto"
    colpali_top_k: int = 3
    colbert_top_k: int = 5


class OptimizationConfig(BaseModel):
    gepa_population_size: int = 8
    gepa_generations: int = 5
    validation_sample_size: int = 10


class CategoryConfig(BaseModel):
    category_name: str
    expected_schema: dict
    extraction_instructions: str
    sample_documents: list[str]
    retrieval: RetrievalConfig = RetrievalConfig()
    optimization: OptimizationConfig = OptimizationConfig()

    @field_validator("sample_documents")
    @classmethod
    def validate_min_two_samples(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 sample documents required for bootstrapping.")
        return v


class AgentRoleConfig(BaseModel):
    model: str | None = None
    text_model: str | None = None
    vision_model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096

    def get_model(self, input_type: str = "text") -> str:
        if input_type == "vision" and self.vision_model:
            return self.vision_model
        if input_type == "text" and self.text_model:
            return self.text_model
        if self.text_model:
            return self.text_model
        if self.model:
            return self.model
        raise ValueError(
            f"No model configured for role with input_type='{input_type}'. "
            f"Set 'model', or 'text_model'/'vision_model'."
        )


class ModelConfig(BaseModel):
    text_model: str | None = None
    vision_model: str | None = None
    agent_roles: dict[str, AgentRoleConfig]


def load_model_config(path: Path | None = None) -> ModelConfig:
    path = path or settings.model_config_path
    with open(path) as f:
        return ModelConfig(**json.load(f))


def load_category_config(
    category_name: str, path: Path | None = None
) -> CategoryConfig:
    if path is None:
        path = settings.configs_dir / "categories" / f"{category_name}.json"
    with open(path) as f:
        return CategoryConfig(**json.load(f))


def list_category_configs() -> list[str]:
    cat_dir = settings.configs_dir / "categories"
    if not cat_dir.exists():
        return []
    return [p.stem for p in cat_dir.glob("*.json")]
