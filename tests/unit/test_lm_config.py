import json
import pytest
from pathlib import Path

from src.config.loader import ModelConfig, CategoryConfig


class TestModelConfigLoading:
    def test_parse_valid_config(self):
        raw = {
            "agent_roles": {
                "extractor": {
                    "model": "openai/gpt-4o-mini",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
                "judge": {
                    "model": "anthropic/claude-sonnet-4-20250514",
                    "temperature": 0.0,
                },
            }
        }
        config = ModelConfig(**raw)
        assert config.agent_roles["extractor"].model == "openai/gpt-4o-mini"
        assert config.agent_roles["judge"].temperature == 0.0

    def test_missing_role_fails(self):
        config = ModelConfig(agent_roles={"extractor": {"model": "openai/gpt-4o-mini"}})
        assert "judge" not in config.agent_roles

    def test_defaults_applied(self):
        from src.config.loader import AgentRoleConfig

        role = AgentRoleConfig(model="openai/gpt-4o")
        assert role.temperature == 0.0
        assert role.max_tokens == 4096


class TestCategoryConfigLoading:
    def test_parse_full_config(self):
        raw = {
            "category_name": "invoice",
            "expected_schema": {
                "type": "object",
                "properties": {"total": {"type": "number"}},
            },
            "extraction_instructions": "Extract invoice total.",
            "sample_documents": ["a.pdf", "b.pdf"],
            "retrieval": {"colpali_top_k": 5},
        }
        config = CategoryConfig(**raw)
        assert config.retrieval.colpali_top_k == 5
        assert config.retrieval.colbert_top_k == 5

    def test_from_json_file(self, tmp_path):
        raw = {
            "category_name": "test",
            "expected_schema": {"type": "object", "properties": {}},
            "extraction_instructions": "test",
            "sample_documents": ["a.pdf", "b.pdf"],
        }
        config_file = tmp_path / "test.json"
        config_file.write_text(json.dumps(raw))

        from src.config.loader import load_category_config

        config = load_category_config("test", path=config_file)
        assert config.category_name == "test"
