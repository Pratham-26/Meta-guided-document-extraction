import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.schemas.trace import TraceEntry


class TestLogTrace:
    def test_writes_single_trace_entry(self, tmp_path, sample_trace_entry, monkeypatch):
        from src.config import settings
        from src.storage import trace_logger

        monkeypatch.setattr(settings, "data_dir", tmp_path)

        trace_logger.log_trace(sample_trace_entry)

        phase_dir = tmp_path / "traces" / "test_category" / "extraction_traces"
        files = list(phase_dir.glob("trace_*.jsonl"))
        assert len(files) == 1

        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["agent_role"] == "extractor"
        assert parsed["phase"] == "extraction"

    def test_writes_to_correct_phase_dir(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.storage import trace_logger

        monkeypatch.setattr(settings, "data_dir", tmp_path)

        entry = TraceEntry(
            timestamp=datetime.now(timezone.utc),
            agent_role="judge",
            phase="evaluation",
            category="test_category",
            prompt="Evaluate...",
            response='{"quality_tier": "high"}',
            model="claude-sonnet",
            provider="anthropic",
            token_usage={},
        )

        trace_logger.log_trace(entry)

        phase_dir = tmp_path / "traces" / "test_category" / "judge_traces"
        files = list(phase_dir.glob("trace_*.jsonl"))
        assert len(files) == 1


class TestLogTraces:
    def test_writes_multiple_entries(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.storage import trace_logger

        monkeypatch.setattr(settings, "data_dir", tmp_path)

        entries = [
            TraceEntry(
                timestamp=datetime.now(timezone.utc),
                agent_role="extractor",
                phase="extraction",
                category="test_category",
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                model="gpt-4o",
                provider="openai",
                token_usage={},
            )
            for i in range(3)
        ]

        trace_logger.log_traces(entries)

        phase_dir = tmp_path / "traces" / "test_category" / "extraction_traces"
        files = list(phase_dir.glob("trace_*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 3

    def test_empty_list_no_writes(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.storage import trace_logger

        monkeypatch.setattr(settings, "data_dir", tmp_path)

        trace_logger.log_traces([])

        trace_root = tmp_path / "traces"
        if trace_root.exists():
            jsonl_files = list(trace_root.rglob("*.jsonl"))
            assert len(jsonl_files) == 0

    def test_groups_by_category_and_phase(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.storage import trace_logger

        monkeypatch.setattr(settings, "data_dir", tmp_path)

        entries = [
            TraceEntry(
                timestamp=datetime.now(timezone.utc),
                agent_role="extractor",
                phase="extraction",
                category="cat_a",
                prompt="p1",
                response="r1",
                model="m",
                provider="p",
                token_usage={},
            ),
            TraceEntry(
                timestamp=datetime.now(timezone.utc),
                agent_role="judge",
                phase="evaluation",
                category="cat_b",
                prompt="p2",
                response="r2",
                model="m",
                provider="p",
                token_usage={},
            ),
        ]

        trace_logger.log_traces(entries)

        cat_a_dir = tmp_path / "traces" / "cat_a" / "extraction_traces"
        cat_b_dir = tmp_path / "traces" / "cat_b" / "judge_traces"
        assert len(list(cat_a_dir.glob("trace_*.jsonl"))) == 1
        assert len(list(cat_b_dir.glob("trace_*.jsonl"))) == 1


class TestReadTraces:
    def test_reads_traces_by_date(self, tmp_path, monkeypatch, sample_trace_entry):
        from src.config import settings
        from src.storage import trace_logger

        monkeypatch.setattr(settings, "data_dir", tmp_path)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trace_logger.log_trace(sample_trace_entry)

        results = trace_logger.read_traces("test_category", "extraction", date=today)
        assert len(results) == 1
        assert results[0].agent_role == "extractor"

    def test_returns_empty_for_nonexistent_date(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.storage import trace_logger

        monkeypatch.setattr(settings, "data_dir", tmp_path)

        results = trace_logger.read_traces(
            "test_category", "extraction", date="2099-01-01"
        )
        assert results == []

    def test_reads_all_traces_without_date(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.storage import trace_logger

        monkeypatch.setattr(settings, "data_dir", tmp_path)

        entry = TraceEntry(
            timestamp=datetime.now(timezone.utc),
            agent_role="extractor",
            phase="extraction",
            category="test_category",
            prompt="p",
            response="r",
            model="m",
            provider="p",
            token_usage={},
        )
        trace_logger.log_trace(entry)

        results = trace_logger.read_traces("test_category", "extraction")
        assert len(results) == 1

    def test_returns_empty_for_nonexistent_phase_dir(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.storage import trace_logger

        monkeypatch.setattr(settings, "data_dir", tmp_path)

        results = trace_logger.read_traces("nonexistent", "extraction")
        assert results == []
