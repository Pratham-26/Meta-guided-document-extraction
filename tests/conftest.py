import pytest
import json
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.schemas.document import DocumentInput, InputType
from src.schemas.gold_standard import GoldStandard
from src.schemas.trace import TraceEntry
from src.schemas.evaluation import JudgeEvaluation, QualityTier, FieldDiff
from src.schemas.question import QuestionSet, QuestionEntry
from src.config.loader import CategoryConfig, ModelConfig, AgentRoleConfig


class FakeDSPyPrediction:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def sample_category_config():
    return CategoryConfig(
        category_name="test_category",
        expected_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["name", "amount"],
        },
        extraction_instructions="Extract name and amount.",
        sample_documents=["./samples/doc1.pdf", "./samples/doc2.pdf"],
    )


@pytest.fixture
def sample_model_config():
    return ModelConfig(
        agent_roles={
            "scout": AgentRoleConfig(
                model="openai/gpt-4o", temperature=0.2, max_tokens=8192
            ),
            "extractor": AgentRoleConfig(model="openai/gpt-4o-mini", temperature=0.0),
            "judge": AgentRoleConfig(
                model="anthropic/claude-sonnet-4-20250514", temperature=0.0
            ),
        }
    )


@pytest.fixture
def sample_document_input():
    return DocumentInput(
        source_uri=Path("test.pdf"),
        input_type=InputType.PDF,
        category="test_category",
    )


@pytest.fixture
def sample_gold_standard():
    return GoldStandard(
        id="gs_001",
        category="test_category",
        input_modality="pdf",
        source_document_uri=Path("sources/test.pdf"),
        extraction={"name": "Acme Corp", "amount": 5000.00},
        approved_by="scout",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_trace_entry():
    return TraceEntry(
        timestamp=datetime.now(timezone.utc),
        agent_role="extractor",
        phase="extraction",
        category="test_category",
        input_modality="pdf",
        prompt="Extract data from...",
        response='{"name": "Acme Corp", "amount": 5000}',
        model="gpt-4o-mini",
        provider="openai",
        token_usage={"prompt_tokens": 100, "completion_tokens": 50},
    )


@pytest.fixture
def sample_judge_evaluation():
    return JudgeEvaluation(
        quality_tier=QualityTier.HIGH,
        feedback="All fields match.",
        field_diffs=[],
        gold_standard_id="gs_001",
        confidence=0.95,
    )


@pytest.fixture
def tmp_category_dir(tmp_path, monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


@pytest.fixture
def mock_dspy_predict():
    with patch("dspy.Predict") as mock_cls:
        yield mock_cls


@pytest.fixture
def mock_dspy_rlm():
    with patch("dspy.RLM") as mock_cls:
        yield mock_cls


@pytest.fixture
def mock_dspy_lm():
    with patch("dspy.LM") as mock_cls:
        lm_instance = MagicMock()
        mock_cls.return_value = lm_instance
        yield lm_instance


@pytest.fixture
def mock_get_lm():
    with patch("src.config.lm.get_lm") as mock:
        fake_lm = MagicMock()
        mock.return_value = fake_lm
        yield mock


@pytest.fixture
def mock_scout_explore():
    with patch("src.agents.scout.agent.ScoutAgent.explore_document") as mock:
        mock.return_value = {
            "exploration": "Found a lease document with landlord Acme Corp and monthly rent of $5000.",
            "extraction": {"name": "Acme Corp", "amount": 5000.00},
        }
        yield mock


@pytest.fixture
def mock_scout_questions():
    with patch(
        "src.agents.scout.agent.ScoutAgent.infer_questions_from_explorations"
    ) as mock:
        mock.return_value = [
            {
                "text": "Who is the landlord?",
                "target_field": "name",
                "retrieval_priority": 1,
            },
            {
                "text": "What is the monthly amount?",
                "target_field": "amount",
                "retrieval_priority": 1,
            },
        ]
        yield mock


@pytest.fixture
def mock_extractor_run():
    with patch("src.agents.extractor.agent.ExtractorAgent.run") as mock:
        mock.return_value = {"name": "Acme Corp", "amount": 5000.00}
        yield mock


@pytest.fixture
def mock_judge_evaluate():
    with patch("src.agents.judge.agent.JudgeAgent.evaluate") as mock:
        mock.return_value = JudgeEvaluation(
            quality_tier=QualityTier.HIGH,
            feedback="All fields match the Gold Standard.",
            field_diffs=[],
            gold_standard_id="gs_001",
            confidence=0.95,
        )
        yield mock


@pytest.fixture
def mock_reflector_analyze():
    with patch("src.optimization.reflector.Reflector.analyze") as mock:
        mock.return_value = {
            "diagnosis": "Missing explicit instruction to capture landlord name from signature block.",
            "suggested_fixes": "Add instruction: 'Look for landlord name in the signature block, not just the header.'",
        }
        yield mock


@pytest.fixture
def mock_mutator_mutate():
    with patch("src.optimization.reflector.PromptMutator.mutate") as mock:
        mock.return_value = {
            "revised_instructions": "Extract name and amount. Look for landlord name in the signature block.",
            "mutation_rationale": "Added instruction to check signature block for names.",
        }
        yield mock


@pytest.fixture
def mock_validate_candidate():
    with patch("src.optimization.gepa.validate_candidate") as mock:
        mock.return_value = {
            "total": 2,
            "low": 0,
            "medium": 1,
            "high": 1,
            "accuracy": 0.5,
            "details": [
                {
                    "gold_standard_id": "gs_001",
                    "quality_tier": "high",
                    "confidence": 0.95,
                },
                {
                    "gold_standard_id": "gs_002",
                    "quality_tier": "medium",
                    "confidence": 0.6,
                },
            ],
        }
        yield mock


@pytest.fixture
def bootstrapped_category(tmp_category_dir):
    from src.storage.paths import ensure_category_dirs
    from src.storage.fs_store import save_gold_standard, save_question_set

    ensure_category_dirs("test_category")

    for i in range(2):
        gs = GoldStandard(
            id=f"gs_{i + 1:03d}",
            category="test_category",
            input_modality="pdf",
            source_document_uri=Path(f"sources/doc_{i + 1}.pdf"),
            extraction={"name": f"Entity {i + 1}", "amount": 1000.0 * (i + 1)},
            approved_by="scout",
            created_at=datetime.now(timezone.utc),
        )
        save_gold_standard("test_category", "pdf", gs)

    qs = QuestionSet(
        category="test_category",
        input_modality="pdf",
        version=1,
        updated_at=datetime.now(timezone.utc).isoformat(),
        questions=[
            QuestionEntry(
                id="q_001", text="Who?", target_field="name", retrieval_priority=1
            ),
            QuestionEntry(
                id="q_002",
                text="How much?",
                target_field="amount",
                retrieval_priority=1,
            ),
        ],
    )
    save_question_set("test_category", "pdf", qs)

    return tmp_category_dir
