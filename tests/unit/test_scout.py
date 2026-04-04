import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.agents.scout.agent import ScoutAgent, ScoutExplore, ScoutQuestionInference
from src.agents.scout.gold_builder import build_and_save
from src.agents.scout.question_store import (
    add_questions,
    get_questions,
    merge_questions,
)
from src.schemas.gold_standard import GoldStandard


class TestScoutAgentExploreDocument:
    def test_returns_exploration_and_extraction(self, mock_dspy_rlm):
        fake_result = MagicMock()
        fake_result.exploration = "Found lease with Acme Corp."
        fake_result.extraction = '{"name": "Acme Corp", "amount": 5000}'
        mock_dspy_rlm.return_value = MagicMock(return_value=fake_result)

        agent = ScoutAgent()
        result = agent.explore_document(
            content="Lease agreement text...",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            instructions="Extract name.",
        )

        assert "exploration" in result
        assert "extraction" in result
        assert result["extraction"]["name"] == "Acme Corp"

    def test_handles_malformed_json_extraction(self, mock_dspy_rlm):
        fake_result = MagicMock()
        fake_result.exploration = "Found something."
        fake_result.extraction = "not valid json"
        mock_dspy_rlm.return_value = MagicMock(return_value=fake_result)

        agent = ScoutAgent()
        result = agent.explore_document("content", {}, "instructions")

        assert result["extraction"]["raw"] == "not valid json"

    def test_with_images_uses_vision_path(self, mock_dspy_rlm):
        fake_result = MagicMock()
        fake_result.exploration = "Visual analysis of PDF."
        fake_result.extraction = '{"name": "Acme Corp", "amount": 5000}'
        mock_dspy_rlm.return_value = MagicMock(return_value=fake_result)

        fake_image = MagicMock()
        vision_lm = MagicMock()

        agent = ScoutAgent(vision_lm=vision_lm)
        with patch(
            "src.agents.scout.agent._encode_images",
            return_value=[{"type": "image", "image": "abc123"}],
        ):
            result = agent.explore_document(
                content="",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                instructions="Extract name.",
                images=[fake_image],
            )

        assert result["extraction"]["name"] == "Acme Corp"

    def test_without_images_uses_text_path(self, mock_dspy_rlm):
        fake_result = MagicMock()
        fake_result.exploration = "Text analysis."
        fake_result.extraction = '{"name": "Acme Corp"}'
        mock_dspy_rlm.return_value = MagicMock(return_value=fake_result)

        agent = ScoutAgent()
        result = agent.explore_document(
            content="some text content",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            instructions="Extract name.",
            images=None,
        )

        assert result["extraction"]["name"] == "Acme Corp"

    def test_vision_lm_restored_after_use(self, mock_dspy_rlm):
        fake_result = MagicMock()
        fake_result.exploration = "Visual."
        fake_result.extraction = "{}"
        mock_dspy_rlm.return_value = MagicMock(return_value=fake_result)

        text_lm = MagicMock()
        vision_lm = MagicMock()

        agent = ScoutAgent(lm=text_lm, vision_lm=vision_lm)
        with patch(
            "src.agents.scout.agent._encode_images",
            return_value=[{"type": "image", "image": "abc"}],
        ):
            agent.explore_document(
                content="",
                schema={},
                instructions="",
                images=[MagicMock()],
            )

        assert mock_dspy_rlm.call_count >= 1


class TestScoutAgentInferQuestions:
    def test_returns_question_list(self, mock_dspy_rlm, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.questions_json = json.dumps(
            [
                {
                    "text": "Who is the landlord?",
                    "target_field": "name",
                    "retrieval_priority": 1,
                },
                {
                    "text": "What is the rent?",
                    "target_field": "amount",
                    "retrieval_priority": 1,
                },
            ]
        )
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = ScoutAgent()
        questions = agent.infer_questions_from_explorations(
            explorations=["Found lease 1", "Found lease 2"],
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            instructions="Extract name.",
        )

        assert len(questions) == 2
        assert questions[0]["target_field"] == "name"

    def test_handles_malformed_json(self, mock_dspy_rlm, mock_dspy_predict):
        fake_result = MagicMock()
        fake_result.questions_json = "not json"
        mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

        agent = ScoutAgent()
        questions = agent.infer_questions_from_explorations([], {}, "")

        assert questions == []


class TestGoldBuilder:
    def test_build_and_save(self, tmp_category_dir):
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_cat")

        gs = build_and_save(
            category="test_cat",
            modality="pdf",
            gs_id="gs_001",
            source_document_uri=Path("sources/test.pdf"),
            extraction={"name": "Acme"},
            approved_by="scout",
        )

        assert gs.id == "gs_001"
        assert gs.approved_by == "scout"

        from src.storage.fs_store import load_gold_standard

        loaded = load_gold_standard("test_cat", "pdf", "gs_001")
        assert loaded.extraction["name"] == "Acme"


class TestQuestionStore:
    def test_add_questions(self, tmp_category_dir):
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_cat")

        qs = add_questions(
            "test_cat",
            "pdf",
            [
                {"text": "Who?", "target_field": "name", "retrieval_priority": 1},
                {
                    "text": "How much?",
                    "target_field": "amount",
                    "retrieval_priority": 2,
                },
            ],
        )

        assert len(qs.questions) == 2
        assert qs.version == 1

    def test_add_questions_appends_to_existing(self, tmp_category_dir):
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_cat")

        add_questions(
            "test_cat",
            "pdf",
            [
                {"text": "Q1?", "target_field": "f1", "retrieval_priority": 1},
            ],
        )
        qs2 = add_questions(
            "test_cat",
            "pdf",
            [
                {"text": "Q2?", "target_field": "f2", "retrieval_priority": 1},
            ],
        )

        assert len(qs2.questions) == 2
        assert qs2.version == 2

    def test_get_questions(self, tmp_category_dir):
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_cat")

        add_questions(
            "test_cat",
            "pdf",
            [
                {"text": "Who?", "target_field": "name", "retrieval_priority": 1},
            ],
        )
        questions = get_questions("test_cat", "pdf")
        assert questions == ["Who?"]

    def test_get_questions_empty(self, tmp_category_dir):
        questions = get_questions("nonexistent", "pdf")
        assert questions == []


class TestMergeQuestions:
    def test_merge_adds_new_fields(self, tmp_category_dir):
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_cat")

        add_questions(
            "test_cat",
            "pdf",
            [
                {"text": "Who?", "target_field": "name", "retrieval_priority": 1},
            ],
        )

        merged = merge_questions(
            "test_cat",
            "pdf",
            [
                {
                    "text": "How much?",
                    "target_field": "amount",
                    "retrieval_priority": 1,
                },
            ],
        )

        assert len(merged.questions) == 2
        assert merged.version == 2

    def test_merge_skips_existing_fields(self, tmp_category_dir):
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_cat")

        add_questions(
            "test_cat",
            "pdf",
            [
                {"text": "Who?", "target_field": "name", "retrieval_priority": 1},
            ],
        )

        merged = merge_questions(
            "test_cat",
            "pdf",
            [
                {
                    "text": "Who is the landlord?",
                    "target_field": "name",
                    "retrieval_priority": 1,
                },
            ],
        )

        assert len(merged.questions) == 1
        assert merged.version == 2
        assert merged.questions[0].text == "Who?"

    def test_merge_creates_new_set_when_none_exists(self, tmp_category_dir):
        from src.storage.paths import ensure_category_dirs

        ensure_category_dirs("test_cat")

        merged = merge_questions(
            "test_cat",
            "pdf",
            [
                {"text": "Who?", "target_field": "name", "retrieval_priority": 1},
                {
                    "text": "How much?",
                    "target_field": "amount",
                    "retrieval_priority": 1,
                },
            ],
        )

        assert len(merged.questions) == 2
        assert merged.version == 1
