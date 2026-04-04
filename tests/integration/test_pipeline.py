import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.orchestration.graph import compile_graph
from src.schemas.document import DocumentInput, InputType
from src.schemas.evaluation import FieldDiff, JudgeEvaluation, QualityTier
from src.schemas.gold_standard import GoldStandard
from src.schemas.question import QuestionEntry, QuestionSet
from src.schemas.trace import TraceEntry
from src.storage.fs_store import (
    save_gold_standard,
    save_question_set,
)
from src.storage.paths import ensure_category_dirs

CATEGORY = "test_lease"
MODALITY = "text"
SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "landlord_name": {"type": "string"},
        "monthly_rent": {"type": "number"},
    },
    "required": ["landlord_name", "monthly_rent"],
}
SAMPLE_INSTRUCTIONS = "Extract the landlord name and monthly rent."


@pytest.fixture
def integration_env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "configs_dir", tmp_path / "configs")
    configs_dir = tmp_path / "configs" / "categories"
    configs_dir.mkdir(parents=True, exist_ok=True)

    cat_config = {
        "category_name": CATEGORY,
        "expected_schema": SAMPLE_SCHEMA,
        "extraction_instructions": SAMPLE_INSTRUCTIONS,
        "sample_documents": ["doc1.txt", "doc2.txt"],
    }
    (configs_dir / f"{CATEGORY}.json").write_text(json.dumps(cat_config))

    ensure_category_dirs(CATEGORY)

    for i in range(2):
        gs = GoldStandard(
            id=f"gs_{i + 1:03d}",
            category=CATEGORY,
            input_modality=MODALITY,
            source_document_uri=Path(f"sources/doc_{i + 1}.txt"),
            extraction={
                "landlord_name": f"Landlord {i + 1}",
                "monthly_rent": 1000.0 * (i + 1),
            },
            approved_by="scout",
            created_at=datetime.now(timezone.utc),
        )
        save_gold_standard(CATEGORY, MODALITY, gs)

    qs = QuestionSet(
        category=CATEGORY,
        input_modality=MODALITY,
        version=1,
        updated_at=datetime.now(timezone.utc).isoformat(),
        questions=[
            QuestionEntry(
                id="q_001",
                text="Who is the landlord?",
                target_field="landlord_name",
                retrieval_priority=1,
            ),
            QuestionEntry(
                id="q_002",
                text="What is the monthly rent?",
                target_field="monthly_rent",
                retrieval_priority=1,
            ),
        ],
    )
    save_question_set(CATEGORY, MODALITY, qs)

    process_config = {"gold_sampling_rate": 100, "auto_gold_initial_count": 0}
    (tmp_path / "configs" / "process_config.json").write_text(
        json.dumps(process_config)
    )

    doc_file = tmp_path / "incoming_lease.txt"
    doc_file.write_text(
        "LEASE AGREEMENT\n\nLandlord: Acme Properties LLC\nMonthly Rent: $5,000"
    )

    return tmp_path, doc_file


class TestRegularPathIntegration:
    @patch("src.orchestration.nodes._build_temp_index")
    @patch("src.orchestration.nodes.get_lm")
    @patch("src.retrieval.colbert.retriever.get_retrieved_chunks")
    def test_regular_text_extraction(
        self, mock_colbert, mock_get_lm, mock_build_tmp, integration_env
    ):
        tmp_path, doc_file = integration_env

        fake_lm = MagicMock()
        mock_get_lm.return_value = fake_lm

        mock_colbert.return_value = [
            {
                "rank": 1,
                "content": "Landlord: Acme Properties LLC\nMonthly Rent: $5,000",
                "document_key": "incoming_lease.txt",
            }
        ]

        with patch("src.agents.extractor.agent.ExtractorAgent.run") as mock_extract:
            mock_extract.return_value = {
                "landlord_name": "Acme Properties LLC",
                "monthly_rent": 5000.0,
            }

            compiled = compile_graph()
            result = compiled.invoke(
                {
                    "category_name": CATEGORY,
                    "input_modality": MODALITY,
                    "document": DocumentInput(
                        source_uri=doc_file,
                        input_type=InputType.TEXT,
                        category=CATEGORY,
                    ),
                }
            )

        assert result.get("error") is None
        assert result["extraction"]["landlord_name"] == "Acme Properties LLC"
        assert result["extraction"]["monthly_rent"] == 5000.0
        assert result["is_gold_doc"] is False
        assert result.get("judge_evaluation") is None

        traces = result.get("trace_entries", [])
        assert len(traces) >= 1
        extractor_traces = [t for t in traces if t.agent_role == "extractor"]
        assert len(extractor_traces) == 1
        assert extractor_traces[0].phase == "extraction"
        assert extractor_traces[0].category == CATEGORY

        trace_dir = tmp_path / "traces" / CATEGORY / MODALITY / "extraction_traces"
        trace_files = list(trace_dir.glob("trace_*.jsonl"))
        assert len(trace_files) == 1
        logged = [
            TraceEntry(**json.loads(line))
            for line in trace_files[0].read_text().strip().split("\n")
            if line.strip()
        ]
        assert len(logged) >= 1
        assert logged[0].agent_role == "extractor"

    @patch("src.orchestration.nodes._build_temp_index")
    @patch("src.orchestration.nodes.get_lm")
    @patch("src.retrieval.colbert.retriever.get_retrieved_chunks")
    def test_regular_path_uses_real_routing(
        self, mock_colbert, mock_get_lm, mock_build_tmp, integration_env
    ):
        tmp_path, doc_file = integration_env
        mock_get_lm.return_value = MagicMock()
        mock_colbert.return_value = [
            {"rank": 1, "content": "some chunk", "document_key": "doc"}
        ]

        with patch(
            "src.agents.extractor.agent.ExtractorAgent.run",
            return_value={"landlord_name": "X", "monthly_rent": 1.0},
        ):
            compiled = compile_graph()
            result = compiled.invoke(
                {
                    "category_name": CATEGORY,
                    "input_modality": MODALITY,
                    "document": DocumentInput(
                        source_uri=doc_file,
                        input_type=InputType.TEXT,
                        category=CATEGORY,
                    ),
                }
            )

        from src.retrieval.router import RetrievalRoute

        assert result["retrieval_route"] == RetrievalRoute.COLBERT
        mock_colbert.assert_called_once()
        call_args = mock_colbert.call_args
        assert call_args[0][0] == CATEGORY
        assert len(call_args[0][1]) == 2


class TestGoldPathIntegration:
    @patch("src.orchestration.nodes._build_temp_index")
    @patch("src.orchestration.nodes.get_lm")
    @patch("src.retrieval.colbert.retriever.get_retrieved_chunks")
    @patch("src.utils.text.clean_text", side_effect=lambda x: x)
    @patch("src.utils.text.truncate_to_tokens", side_effect=lambda x, **kw: x)
    def test_gold_path_with_scout_and_judge(
        self,
        mock_truncate,
        mock_clean,
        mock_colbert,
        mock_get_lm,
        mock_build_tmp,
        integration_env,
    ):
        tmp_path, doc_file = integration_env

        fake_lm = MagicMock()
        mock_get_lm.return_value = fake_lm

        mock_colbert.return_value = [
            {
                "rank": 1,
                "content": "Landlord: Acme Properties LLC\nMonthly Rent: $5,000",
                "document_key": "incoming_lease.txt",
            }
        ]

        with (
            patch("src.agents.scout.agent.ScoutAgent.explore_document") as mock_explore,
            patch(
                "src.agents.scout.agent.ScoutAgent.infer_questions_from_explorations"
            ) as mock_infer,
            patch("src.agents.extractor.agent.ExtractorAgent.run") as mock_extract,
            patch("src.agents.judge.agent.JudgeAgent.evaluate") as mock_judge,
        ):
            mock_explore.return_value = {
                "exploration": "Found lease with Acme Properties LLC as landlord, $5000 rent.",
                "extraction": {
                    "landlord_name": "Acme Properties LLC",
                    "monthly_rent": 5000.0,
                },
            }
            mock_infer.return_value = [
                {
                    "text": "Who is the lessor?",
                    "target_field": "landlord_name",
                    "retrieval_priority": 2,
                },
            ]
            mock_extract.return_value = {
                "landlord_name": "Acme Properties LLC",
                "monthly_rent": 5000.0,
            }
            mock_judge.return_value = JudgeEvaluation(
                quality_tier=QualityTier.HIGH,
                feedback="All fields match perfectly.",
                field_diffs=[],
                gold_standard_id="gs_003",
                confidence=0.98,
            )

            compiled = compile_graph()
            result = compiled.invoke(
                {
                    "category_name": CATEGORY,
                    "input_modality": MODALITY,
                    "document": DocumentInput(
                        source_uri=doc_file,
                        input_type=InputType.TEXT,
                        category=CATEGORY,
                    ),
                    "is_gold_doc": True,
                    "gold_source": "user_flag",
                }
            )

        assert result.get("error") is None
        assert result["extraction"]["landlord_name"] == "Acme Properties LLC"
        assert result["is_gold_doc"] is True
        assert result["gold_source"] == "user_flag"

        assert result["judge_evaluation"] is not None
        assert result["judge_evaluation"].quality_tier == QualityTier.HIGH
        assert result["judge_evaluation"].confidence == 0.98
        assert result["judge_evaluation"].gold_standard_id == "gs_003"

        traces = result.get("trace_entries", [])
        assert len(traces) == 2
        roles = [t.agent_role for t in traces]
        assert "extractor" in roles
        assert "judge" in roles

        trace_dir = tmp_path / "traces" / CATEGORY / MODALITY
        ext_traces = list((trace_dir / "extraction_traces").glob("trace_*.jsonl"))
        judge_traces = list((trace_dir / "judge_traces").glob("trace_*.jsonl"))
        assert len(ext_traces) == 1
        assert len(judge_traces) == 1

        gs_dir = tmp_path / "categories" / CATEGORY / MODALITY / "gold_standards"
        gs_files = list(gs_dir.glob("gs_*.json"))
        assert len(gs_files) == 3
        gs_ids = sorted([f.stem for f in gs_files])
        assert "gs_003" in gs_ids

        from src.storage.fs_store import load_question_set

        qs = load_question_set(CATEGORY, MODALITY)
        assert qs is not None
        assert qs.version == 2
        fields = {q.target_field for q in qs.questions}
        assert "landlord_name" in fields
        assert "monthly_rent" in fields

    @patch("src.orchestration.nodes._build_temp_index")
    @patch("src.orchestration.nodes.get_lm")
    @patch("src.retrieval.colbert.retriever.get_retrieved_chunks")
    def test_gold_path_judge_receives_correct_gold_standard(
        self, mock_colbert, mock_get_lm, mock_build_tmp, integration_env
    ):
        tmp_path, doc_file = integration_env
        mock_get_lm.return_value = MagicMock()
        mock_colbert.return_value = [
            {"rank": 1, "content": "chunk", "document_key": "doc"}
        ]

        judge_calls = []

        def capture_judge(extraction, gold_standard, schema, gold_standard_id):
            judge_calls.append(
                {
                    "extraction": extraction,
                    "gold_standard": gold_standard,
                    "gold_standard_id": gold_standard_id,
                }
            )
            return JudgeEvaluation(
                quality_tier=QualityTier.HIGH,
                feedback="ok",
                field_diffs=[],
                gold_standard_id=gold_standard_id,
                confidence=0.9,
            )

        with (
            patch(
                "src.agents.scout.agent.ScoutAgent.explore_document",
                return_value={
                    "exploration": "explored",
                    "extraction": {"landlord_name": "X", "monthly_rent": 1.0},
                },
            ),
            patch(
                "src.agents.scout.agent.ScoutAgent.infer_questions_from_explorations",
                return_value=[],
            ),
            patch(
                "src.agents.extractor.agent.ExtractorAgent.run",
                return_value={"landlord_name": "X", "monthly_rent": 1.0},
            ),
            patch(
                "src.agents.judge.agent.JudgeAgent.evaluate",
                side_effect=capture_judge,
            ),
        ):
            compiled = compile_graph()
            result = compiled.invoke(
                {
                    "category_name": CATEGORY,
                    "input_modality": MODALITY,
                    "document": DocumentInput(
                        source_uri=doc_file,
                        input_type=InputType.TEXT,
                        category=CATEGORY,
                    ),
                    "is_gold_doc": True,
                    "gold_source": "auto_initial",
                }
            )

        assert result.get("error") is None
        assert len(judge_calls) == 1
        assert judge_calls[0]["gold_standard_id"].startswith("gs_")
        assert isinstance(judge_calls[0]["gold_standard"], dict)
        assert isinstance(judge_calls[0]["extraction"], dict)


class TestContextGateIntegration:
    def test_halt_on_missing_context(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        monkeypatch.setattr(settings, "configs_dir", tmp_path / "configs")
        configs_dir = tmp_path / "configs" / "categories"
        configs_dir.mkdir(parents=True, exist_ok=True)

        cat_config = {
            "category_name": "missing_cat",
            "expected_schema": {"type": "object", "properties": {}},
            "extraction_instructions": "none",
            "sample_documents": ["a.txt", "b.txt"],
        }
        (configs_dir / "missing_cat.json").write_text(json.dumps(cat_config))

        compiled = compile_graph()
        result = compiled.invoke(
            {
                "category_name": "missing_cat",
                "input_modality": "text",
                "document": DocumentInput(
                    source_uri=Path("fake.txt"),
                    input_type=InputType.TEXT,
                    category="missing_cat",
                ),
            }
        )

        assert result.get("error") is not None
        assert "No Scout context" in result["error"]
        assert result.get("extraction") is None


class TestAutoGoldIntegration:
    @patch("src.orchestration.nodes._build_temp_index")
    @patch("src.orchestration.nodes.get_lm")
    @patch("src.retrieval.colbert.retriever.get_retrieved_chunks")
    def test_auto_gold_first_document(
        self, mock_colbert, mock_get_lm, mock_build_tmp, integration_env
    ):
        tmp_path, doc_file = integration_env

        process_config = {"gold_sampling_rate": 100, "auto_gold_initial_count": 5}
        (tmp_path / "configs" / "process_config.json").write_text(
            json.dumps(process_config)
        )

        mock_get_lm.return_value = MagicMock()
        mock_colbert.return_value = [
            {"rank": 1, "content": "chunk", "document_key": "doc"}
        ]

        with (
            patch(
                "src.agents.scout.agent.ScoutAgent.explore_document",
                return_value={
                    "exploration": "explored",
                    "extraction": {"landlord_name": "A", "monthly_rent": 1.0},
                },
            ),
            patch(
                "src.agents.scout.agent.ScoutAgent.infer_questions_from_explorations",
                return_value=[],
            ),
            patch(
                "src.agents.extractor.agent.ExtractorAgent.run",
                return_value={"landlord_name": "A", "monthly_rent": 1.0},
            ),
            patch(
                "src.agents.judge.agent.JudgeAgent.evaluate",
                return_value=JudgeEvaluation(
                    quality_tier=QualityTier.HIGH,
                    feedback="ok",
                    field_diffs=[],
                    gold_standard_id="gs_003",
                    confidence=0.9,
                ),
            ),
        ):
            compiled = compile_graph()
            result = compiled.invoke(
                {
                    "category_name": CATEGORY,
                    "input_modality": MODALITY,
                    "document": DocumentInput(
                        source_uri=doc_file,
                        input_type=InputType.TEXT,
                        category=CATEGORY,
                    ),
                }
            )

        assert result.get("error") is None
        assert result["is_gold_doc"] is True
        assert result["gold_source"] == "auto_initial"

        counter_path = (
            tmp_path / "categories" / CATEGORY / MODALITY / "sampling_counter.json"
        )
        assert counter_path.exists()
        counter = json.loads(counter_path.read_text())
        assert counter["total"] == 1


class TestRealConfigLoading:
    @patch("src.orchestration.nodes._build_temp_index")
    @patch("src.orchestration.nodes.get_lm")
    @patch("src.retrieval.colbert.retriever.get_retrieved_chunks")
    def test_resolve_config_loads_real_category_config(
        self, mock_colbert, mock_get_lm, mock_build_tmp, integration_env
    ):
        tmp_path, doc_file = integration_env
        mock_get_lm.return_value = MagicMock()
        mock_colbert.return_value = [
            {"rank": 1, "content": "chunk", "document_key": "doc"}
        ]

        with patch(
            "src.agents.extractor.agent.ExtractorAgent.run",
            return_value={"landlord_name": "X", "monthly_rent": 1.0},
        ):
            compiled = compile_graph()
            result = compiled.invoke(
                {
                    "category_name": CATEGORY,
                    "input_modality": MODALITY,
                    "document": DocumentInput(
                        source_uri=doc_file,
                        input_type=InputType.TEXT,
                        category=CATEGORY,
                    ),
                }
            )

        assert result.get("error") is None
        assert result["schema"] == SAMPLE_SCHEMA
        assert result["instructions"] == SAMPLE_INSTRUCTIONS


class TestPDFRoutingIntegration:
    @patch("src.orchestration.nodes._build_temp_index")
    @patch("src.orchestration.nodes.get_lm")
    @patch("src.retrieval.colpali.retriever.get_retrieved_pages")
    def test_pdf_routes_to_colpali(
        self, mock_colpali, mock_get_lm, mock_build_tmp, integration_env
    ):
        tmp_path, doc_file = integration_env

        pdf_dir = tmp_path / "categories" / CATEGORY / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        gs_dir = pdf_dir / "gold_standards"
        gs_dir.mkdir(parents=True, exist_ok=True)

        gs = GoldStandard(
            id="gs_001",
            category=CATEGORY,
            input_modality="pdf",
            source_document_uri=Path("sources/doc.pdf"),
            extraction={"landlord_name": "LLC", "monthly_rent": 3000.0},
            approved_by="scout",
            created_at=datetime.now(timezone.utc),
        )
        save_gold_standard(CATEGORY, "pdf", gs)

        qs = QuestionSet(
            category=CATEGORY,
            input_modality="pdf",
            version=1,
            updated_at=datetime.now(timezone.utc).isoformat(),
            questions=[
                QuestionEntry(
                    id="q_001",
                    text="Who is the landlord?",
                    target_field="landlord_name",
                    retrieval_priority=1,
                ),
            ],
        )
        save_question_set(CATEGORY, "pdf", qs)

        mock_get_lm.return_value = MagicMock()
        mock_colpali.return_value = [{"page_number": 1, "content": "page 1 text"}]

        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        with patch(
            "src.agents.extractor.agent.ExtractorAgent.run",
            return_value={"landlord_name": "LLC", "monthly_rent": 3000.0},
        ):
            compiled = compile_graph()
            result = compiled.invoke(
                {
                    "category_name": CATEGORY,
                    "input_modality": "pdf",
                    "document": DocumentInput(
                        source_uri=fake_pdf,
                        input_type=InputType.PDF,
                        category=CATEGORY,
                    ),
                }
            )

        from src.retrieval.router import RetrievalRoute

        assert result.get("error") is None
        assert result["retrieval_route"] == RetrievalRoute.COLPALI
        mock_colpali.assert_called_once()
