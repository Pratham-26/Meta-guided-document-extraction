import json
from datetime import datetime, timezone
from pathlib import Path

from src.schemas.gold_standard import GoldStandard
from src.schemas.category import QuestionSet, QuestionEntry
from src.storage.fs_store import (
    save_gold_standard,
    load_gold_standard,
    list_gold_standards,
    delete_gold_standard,
    has_context,
    save_question_set,
    load_question_set,
)
from src.storage.paths import ensure_category_dirs


class TestGoldStandardCRUD:
    def test_save_and_load(self, tmp_category_dir):
        ensure_category_dirs("test_cat")
        gs = GoldStandard(
            id="gs_001",
            category="test_cat",
            source_document_uri=Path("test.pdf"),
            extraction={"name": "Test"},
            approved_by="scout",
            created_at=datetime.now(timezone.utc),
        )
        save_gold_standard("test_cat", gs)
        loaded = load_gold_standard("test_cat", "gs_001")
        assert loaded.id == "gs_001"
        assert loaded.extraction["name"] == "Test"

    def test_list_gold_standards(self, tmp_category_dir):
        ensure_category_dirs("test_cat")
        for i in range(3):
            gs = GoldStandard(
                id=f"gs_{i:03d}",
                category="test_cat",
                source_document_uri=Path("test.pdf"),
                extraction={"idx": i},
                approved_by="scout",
                created_at=datetime.now(timezone.utc),
            )
            save_gold_standard("test_cat", gs)

        results = list_gold_standards("test_cat")
        assert len(results) == 3

    def test_delete_gold_standard(self, tmp_category_dir):
        ensure_category_dirs("test_cat")
        gs = GoldStandard(
            id="gs_del",
            category="test_cat",
            source_document_uri=Path("test.pdf"),
            extraction={},
            approved_by="scout",
            created_at=datetime.now(timezone.utc),
        )
        save_gold_standard("test_cat", gs)
        assert delete_gold_standard("test_cat", "gs_del")
        assert not delete_gold_standard("test_cat", "gs_del")

    def test_list_empty_category(self, tmp_category_dir):
        results = list_gold_standards("nonexistent")
        assert results == []


class TestQuestionStore:
    def test_save_and_load(self, tmp_category_dir):
        ensure_category_dirs("test_cat")
        qs = QuestionSet(
            category="test_cat",
            version=1,
            updated_at=datetime.now(timezone.utc).isoformat(),
            questions=[
                QuestionEntry(
                    id="q_001", text="Who?", target_field="name", retrieval_priority=1
                ),
            ],
        )
        save_question_set("test_cat", qs)
        loaded = load_question_set("test_cat")
        assert loaded is not None
        assert len(loaded.questions) == 1
        assert loaded.questions[0].text == "Who?"

    def test_load_nonexistent(self, tmp_category_dir):
        result = load_question_set("nonexistent")
        assert result is None


class TestHasContext:
    def test_no_context(self, tmp_category_dir):
        assert not has_context("empty_cat")

    def test_with_context(self, tmp_category_dir):
        ensure_category_dirs("test_cat")
        gs = GoldStandard(
            id="gs_001",
            category="test_cat",
            source_document_uri=Path("test.pdf"),
            extraction={},
            approved_by="scout",
            created_at=datetime.now(timezone.utc),
        )
        save_gold_standard("test_cat", gs)

        qs = QuestionSet(
            category="test_cat",
            version=1,
            updated_at=datetime.now(timezone.utc).isoformat(),
            questions=[QuestionEntry(id="q_001", text="What?", target_field="x")],
        )
        save_question_set("test_cat", qs)

        assert has_context("test_cat")
