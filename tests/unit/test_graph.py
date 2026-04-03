from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from src.orchestration.graph import (
    build_graph,
    compile_graph,
    _has_error,
    _route_retrieval,
)
from src.orchestration.state import PipelineState
from src.schemas.document import DocumentInput, InputType
from src.schemas.trace import TraceEntry
from src.retrieval.router import RetrievalRoute


EXPECTED_NODES = [
    "check_context",
    "resolve_config",
    "load_questions",
    "route_input",
    "retrieve",
    "extract",
    "judge",
    "log_traces",
]


class TestBuildGraph:
    def test_graph_has_all_nodes(self):
        graph = build_graph()
        for name in EXPECTED_NODES:
            assert name in graph.nodes, f"Missing node: {name}"

    def test_graph_starts_with_check_context(self):
        graph = build_graph()
        start_edges = [e for e in graph.edges if e[0] == "__start__"]
        assert len(start_edges) == 1
        assert start_edges[0][1] == "check_context"


class TestHasError:
    def test_returns_halt_on_error(self):
        state: PipelineState = {"error": "something failed"}
        assert _has_error(state) == "halt"

    def test_returns_continue_without_error(self):
        state: PipelineState = {"category_name": "leases"}
        assert _has_error(state) == "continue"

    def test_returns_continue_on_empty_state(self):
        state: PipelineState = {}
        assert _has_error(state) == "continue"


class TestRouteRetrieval:
    def test_routes_to_colpali(self):
        state: PipelineState = {"retrieval_route": RetrievalRoute.COLPALI}
        assert _route_retrieval(state) == "colpali"

    def test_routes_to_colbert(self):
        state: PipelineState = {"retrieval_route": RetrievalRoute.COLBERT}
        assert _route_retrieval(state) == "colbert"

    def test_defaults_to_colbert(self):
        state: PipelineState = {}
        assert _route_retrieval(state) == "colbert"


class TestCompileGraph:
    def test_compile_returns_compiled_graph(self):
        compiled = compile_graph()
        assert compiled is not None
        assert hasattr(compiled, "invoke")


class TestIntegration:
    def test_happy_path_pipeline(self, sample_document_input):
        trace_extract = TraceEntry(
            timestamp=datetime.now(timezone.utc),
            agent_role="extractor",
            phase="extraction",
            category="test",
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
        p3 = patch(
            "src.orchestration.nodes.load_questions", side_effect=load_questions_fn
        )
        p4 = patch("src.orchestration.nodes.route_input", side_effect=route_input_fn)
        p5 = patch("src.orchestration.nodes.retrieve", side_effect=retrieve_fn)
        p6 = patch("src.orchestration.nodes.extract", side_effect=extract_fn)
        p7 = patch("src.orchestration.nodes.judge", side_effect=judge_fn)
        p8 = patch("src.orchestration.nodes.log_traces", side_effect=log_traces_fn)

        with p1, p2, p3, p4, p5, p6, p7, p8:
            compiled = compile_graph()
            result = compiled.invoke(
                {
                    "category_name": "test",
                    "document": sample_document_input,
                }
            )

        assert result["extraction"] == {"name": "Test", "amount": 100}
        assert result["judge_evaluation"] == {"quality": "high"}
        assert len(result["trace_entries"]) == 2

    def test_error_halts_pipeline(self, sample_document_input):
        extract_called = False
        judge_called = False

        def check_context_error(state):
            return {**state, "error": "No context found"}

        def extract_fn(state):
            nonlocal extract_called
            extract_called = True
            return state

        def judge_fn(state):
            nonlocal judge_called
            judge_called = True
            return state

        p1 = patch(
            "src.orchestration.nodes.check_context", side_effect=check_context_error
        )
        p2 = patch("src.orchestration.nodes.resolve_config", side_effect=lambda s: s)
        p3 = patch("src.orchestration.nodes.load_questions", side_effect=lambda s: s)
        p4 = patch("src.orchestration.nodes.route_input", side_effect=lambda s: s)
        p5 = patch("src.orchestration.nodes.retrieve", side_effect=lambda s: s)
        p6 = patch("src.orchestration.nodes.extract", side_effect=extract_fn)
        p7 = patch("src.orchestration.nodes.judge", side_effect=judge_fn)
        p8 = patch("src.orchestration.nodes.log_traces", side_effect=lambda s: s)

        with p1, p2, p3, p4, p5, p6, p7, p8:
            compiled = compile_graph()
            result = compiled.invoke(
                {
                    "category_name": "test",
                    "document": sample_document_input,
                }
            )

        assert "error" in result
        assert not extract_called
        assert not judge_called
