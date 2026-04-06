from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from src.orchestration.graph import run_pipeline
from src.orchestration.state import PipelineState
from src.schemas.document import DocumentInput, InputType
from src.schemas.trace import TraceEntry
from src.retrieval.router import RetrievalRoute

EXPECTED_STEPS = [
    "check_context",
    "resolve_config",
    "detect_gold",
    "run_scout_for_gold",
    "load_questions",
    "route_input",
    "retrieve",
    "extract",
    "judge",
    "log_traces",
]


class TestRunPipeline:
    def test_regular_path_pipeline(self, sample_document_input):
        trace_extract = TraceEntry(
            timestamp=datetime.now(timezone.utc),
            agent_role="extractor",
            phase="extraction",
            category="test",
            input_modality="pdf",
            prompt="ctx",
            response="extracted",
            model="m",
            provider="p",
            token_usage={},
        )

        def check_context_fn(state):
            return state

        def resolve_config_fn(state):
            return {**state, "schema": {}, "instructions": "do it"}

        def detect_gold_fn(state):
            return {**state, "is_gold_doc": False, "gold_source": None}

        def load_questions_fn(state):
            return {**state, "questions": ["q1?"]}

        def route_input_fn(state):
            return {**state, "retrieval_route": RetrievalRoute.COLBERT}

        def retrieve_fn(state):
            return {**state, "retrieved_context": "some context"}

        def extract_fn(state):
            return {
                **state,
                "extraction": {"name": "Test", "amount": 100},
                "trace_entries": state.get("trace_entries", []) + [trace_extract],
            }

        def judge_fn(state):
            return state

        def log_traces_fn(state):
            return state

        p1 = patch(
            "src.orchestration.nodes.check_context", side_effect=check_context_fn
        )
        p2 = patch(
            "src.orchestration.nodes.resolve_config", side_effect=resolve_config_fn
        )
        p3 = patch("src.orchestration.nodes.detect_gold", side_effect=detect_gold_fn)
        p4 = patch(
            "src.orchestration.nodes.run_scout_for_gold",
            side_effect=lambda s: s,
        )
        p5 = patch(
            "src.orchestration.nodes.load_questions", side_effect=load_questions_fn
        )
        p6 = patch("src.orchestration.nodes.route_input", side_effect=route_input_fn)
        p7 = patch("src.orchestration.nodes.retrieve", side_effect=retrieve_fn)
        p8 = patch("src.orchestration.nodes.extract", side_effect=extract_fn)
        p9 = patch("src.orchestration.nodes.judge", side_effect=judge_fn)
        p10 = patch("src.orchestration.nodes.log_traces", side_effect=log_traces_fn)

        with p1, p2, p3, p4, p5, p6, p7, p8, p9, p10:
            result = run_pipeline(
                {
                    "category_name": "test",
                    "input_modality": "pdf",
                    "document": sample_document_input,
                }
            )

        assert result["extraction"] == {"name": "Test", "amount": 100}
        assert len(result["trace_entries"]) == 1

    def test_gold_path_pipeline(self, sample_document_input):
        trace_extract = TraceEntry(
            timestamp=datetime.now(timezone.utc),
            agent_role="extractor",
            phase="extraction",
            category="test",
            input_modality="pdf",
            prompt="ctx",
            response="extracted",
            model="m",
            provider="p",
            token_usage={},
        )
        trace_judge = TraceEntry(
            timestamp=datetime.now(timezone.utc),
            agent_role="judge",
            phase="evaluation",
            category="test",
            input_modality="pdf",
            prompt="extraction",
            response="eval",
            model="m",
            provider="p",
            token_usage={},
        )

        def check_context_fn(state):
            return state

        def resolve_config_fn(state):
            return {**state, "schema": {}, "instructions": "do it"}

        def detect_gold_fn(state):
            return {**state, "is_gold_doc": True, "gold_source": "user_flag"}

        def run_scout_fn(state):
            return state

        def load_questions_fn(state):
            return {**state, "questions": ["q1?"]}

        def route_input_fn(state):
            return {**state, "retrieval_route": RetrievalRoute.COLBERT}

        def retrieve_fn(state):
            return {**state, "retrieved_context": "some context"}

        def extract_fn(state):
            return {
                **state,
                "extraction": {"name": "Test", "amount": 100},
                "trace_entries": state.get("trace_entries", []) + [trace_extract],
            }

        def judge_fn(state):
            return {
                **state,
                "judge_evaluation": {"quality": "high"},
                "trace_entries": state.get("trace_entries", []) + [trace_judge],
            }

        def log_traces_fn(state):
            return state

        p1 = patch(
            "src.orchestration.nodes.check_context", side_effect=check_context_fn
        )
        p2 = patch(
            "src.orchestration.nodes.resolve_config", side_effect=resolve_config_fn
        )
        p3 = patch("src.orchestration.nodes.detect_gold", side_effect=detect_gold_fn)
        p4 = patch(
            "src.orchestration.nodes.run_scout_for_gold", side_effect=run_scout_fn
        )
        p5 = patch(
            "src.orchestration.nodes.load_questions", side_effect=load_questions_fn
        )
        p6 = patch("src.orchestration.nodes.route_input", side_effect=route_input_fn)
        p7 = patch("src.orchestration.nodes.retrieve", side_effect=retrieve_fn)
        p8 = patch("src.orchestration.nodes.extract", side_effect=extract_fn)
        p9 = patch("src.orchestration.nodes.judge", side_effect=judge_fn)
        p10 = patch("src.orchestration.nodes.log_traces", side_effect=log_traces_fn)

        with p1, p2, p3, p4, p5, p6, p7, p8, p9, p10:
            result = run_pipeline(
                {
                    "category_name": "test",
                    "input_modality": "pdf",
                    "document": sample_document_input,
                    "is_gold_doc": True,
                    "gold_source": "user_flag",
                }
            )

        assert result["extraction"] == {"name": "Test", "amount": 100}
        assert result["judge_evaluation"] == {"quality": "high"}
        assert len(result["trace_entries"]) == 2

    def test_error_halts_pipeline(self, sample_document_input):
        extract_called = False

        def check_context_error(state):
            return {**state, "error": "No context found"}

        def extract_fn(state):
            nonlocal extract_called
            extract_called = True
            return state

        p1 = patch(
            "src.orchestration.nodes.check_context", side_effect=check_context_error
        )
        p8 = patch("src.orchestration.nodes.extract", side_effect=extract_fn)

        with p1, p8:
            result = run_pipeline(
                {
                    "category_name": "test",
                    "input_modality": "pdf",
                    "document": sample_document_input,
                }
            )

        assert "error" in result
        assert not extract_called
