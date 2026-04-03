import json
import pytest
from pathlib import Path

from src.config.loader import ModelConfig, CategoryConfig, AgentRoleConfig


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
        role = AgentRoleConfig(model="openai/gpt-4o")
        assert role.temperature == 0.0
        assert role.max_tokens == 4096

    def test_parse_dual_model_config(self):
        raw = {
            "text_model": "zai/glm-4.7-flash",
            "vision_model": "zai/glm-4.6v-flash",
            "agent_roles": {
                "scout": {
                    "text_model": "zai/glm-4.7-flash",
                    "vision_model": "zai/glm-4.6v-flash",
                    "temperature": 0.2,
                    "max_tokens": 8192,
                },
                "extractor": {
                    "text_model": "zai/glm-4.7-flash",
                    "vision_model": "zai/glm-4.6v-flash",
                    "temperature": 0.0,
                },
                "judge": {
                    "model": "zai/glm-4.7-flash",
                    "temperature": 0.0,
                },
            },
        }
        config = ModelConfig(**raw)
        assert config.text_model == "zai/glm-4.7-flash"
        assert config.vision_model == "zai/glm-4.6v-flash"
        assert config.agent_roles["scout"].text_model == "zai/glm-4.7-flash"
        assert config.agent_roles["scout"].vision_model == "zai/glm-4.6v-flash"
        assert config.agent_roles["judge"].model == "zai/glm-4.7-flash"


class TestAgentRoleConfigGetModel:
    def test_single_model_returns_for_both_types(self):
        role = AgentRoleConfig(model="openai/gpt-4o")
        assert role.get_model("text") == "openai/gpt-4o"
        assert role.get_model("vision") == "openai/gpt-4o"

    def test_dual_model_selects_correct(self):
        role = AgentRoleConfig(
            text_model="zai/glm-4.7-flash",
            vision_model="zai/glm-4.6v-flash",
        )
        assert role.get_model("text") == "zai/glm-4.7-flash"
        assert role.get_model("vision") == "zai/glm-4.6v-flash"

    def test_vision_falls_back_to_text_model(self):
        role = AgentRoleConfig(
            text_model="zai/glm-4.7-flash",
        )
        assert role.get_model("vision") == "zai/glm-4.7-flash"
        assert role.get_model("text") == "zai/glm-4.7-flash"

    def test_text_falls_back_to_single_model(self):
        role = AgentRoleConfig(
            model="openai/gpt-4o",
        )
        assert role.get_model("text") == "openai/gpt-4o"
        assert role.get_model("vision") == "openai/gpt-4o"

    def test_no_model_raises(self):
        role = AgentRoleConfig()
        with pytest.raises(ValueError, match="No model configured"):
            role.get_model("text")

    def test_dual_model_falls_back_to_single(self):
        role = AgentRoleConfig(
            model="openai/gpt-4o",
            text_model="zai/glm-4.7-flash",
            vision_model="zai/glm-4.6v-flash",
        )
        assert role.get_model("text") == "zai/glm-4.7-flash"
        assert role.get_model("vision") == "zai/glm-4.6v-flash"


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
