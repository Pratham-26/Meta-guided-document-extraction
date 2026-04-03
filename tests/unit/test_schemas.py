import pytest
from pathlib import Path
from datetime import datetime, timezone

from src.schemas.document import DocumentInput, InputType
from src.schemas.gold_standard import GoldStandard
from src.schemas.trace import TraceEntry
from src.schemas.evaluation import JudgeEvaluation, QualityTier, FieldDiff
from src.config.loader import CategoryConfig, ModelConfig


class TestDocumentInput:
    def test_pdf_input(self):
        doc = DocumentInput(
            source_uri=Path("test.pdf"), input_type=InputType.PDF, category="test"
        )
        assert doc.input_type == InputType.PDF
        assert doc.raw_text is None
        assert doc.metadata == {}

    def test_text_input_with_raw_text(self):
        doc = DocumentInput(
            source_uri=Path("test.txt"),
            input_type=InputType.TEXT,
            category="test",
            raw_text="Hello world",
        )
        assert doc.input_type == InputType.TEXT
        assert doc.raw_text == "Hello world"


class TestGoldStandard:
    def test_basic_creation(self):
        gs = GoldStandard(
            id="gs_001",
            category="test",
            source_document_uri=Path("test.pdf"),
            extraction={"name": "Acme"},
            approved_by="scout",
            created_at=datetime.now(timezone.utc),
        )
        assert gs.id == "gs_001"
        assert gs.supersedes is None

    def test_with_supersession(self):
        gs = GoldStandard(
            id="gs_002",
            category="test",
            source_document_uri=Path("test.pdf"),
            extraction={"name": "Acme"},
            approved_by="human",
            created_at=datetime.now(timezone.utc),
            supersedes="gs_001",
        )
        assert gs.supersedes == "gs_001"


class TestTraceEntry:
    def test_basic_creation(self, sample_trace_entry):
        assert sample_trace_entry.agent_role == "extractor"
        assert sample_trace_entry.quality_tier is None
        assert sample_trace_entry.document_id is None


class TestJudgeEvaluation:
    def test_high_quality(self, sample_judge_evaluation):
        assert sample_judge_evaluation.quality_tier == QualityTier.HIGH
        assert sample_judge_evaluation.confidence == 0.95

    def test_field_diffs(self):
        diff = FieldDiff(
            field="amount", expected="5000", actual="500", issue="Value mismatch"
        )
        assert diff.issue == "Value mismatch"


class TestCategoryConfig:
    def test_defaults(self, sample_category_config):
        assert sample_category_config.retrieval.colpali_top_k == 3
        assert sample_category_config.optimization.gepa_population_size == 8

    def test_minimum_sample_documents(self):
        with pytest.raises(Exception):
            CategoryConfig(
                category_name="bad",
                expected_schema={"type": "object", "properties": {}},
                extraction_instructions="test",
                sample_documents=["only_one.pdf"],
            )


class TestModelConfig:
    def test_parse_model_config(self, sample_model_config):
        assert "scout" in sample_model_config.agent_roles
        assert sample_model_config.agent_roles["scout"].model == "openai/gpt-4o"
