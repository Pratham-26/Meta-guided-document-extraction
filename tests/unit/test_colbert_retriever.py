from pathlib import Path

import pytest


class TestBuildIndex:
    def test_creates_index_from_documents(self, tmp_category_dir, tmp_path):
        doc_file = tmp_path / "doc1.txt"
        doc_file.write_text(
            "Sample document content about service agreements.", encoding="utf-8"
        )

        from src.retrieval.colbert.indexer import build_index

        result = build_index("test_category", [doc_file])

        assert result is not None
        assert len(result["chunks"]) > 0

    def test_handles_string_documents(self, tmp_category_dir, tmp_path):
        text_file = tmp_path / "doc.txt"
        text_file.write_text(
            "Plain text document content about compensation fees.", encoding="utf-8"
        )

        from src.retrieval.colbert.indexer import build_index

        result = build_index("test_category", [text_file])

        assert result is not None
        assert len(result["chunks"]) > 0


class TestRetrieve:
    def test_retrieves_and_ranks(self, tmp_category_dir, tmp_path):
        doc_file = tmp_path / "doc1.txt"
        doc_file.write_text(
            "The agreement number is SA-001. "
            "Compensation is hourly at $150. "
            "The term is 12 months.",
            encoding="utf-8",
        )

        from src.retrieval.colbert.indexer import build_index, retrieve

        build_index("test_category", [doc_file])
        result = retrieve("test_category", ["agreement number"], top_k=5)

        assert len(result) >= 1
        assert "content" in result[0]
        assert "score" in result[0]

    def test_respects_top_k(self, tmp_category_dir, tmp_path):
        doc_file = tmp_path / "doc1.txt"
        doc_file.write_text(
            "Agreement SA-001. Compensation fees hourly rate $150. "
            "Service provider is Acme Corp. Client is Beta Inc. "
            "Term duration 12 months. Liability cap $50000. "
            "Governing law is Delaware. Dispute resolution by arbitration. "
            "Insurance general liability $1M. Warranty period 6 months. "
            "Confidentiality 3 years. Non-solicitation 12 months. "
            "Intellectual property owned by client. Termination 30 days notice.",
            encoding="utf-8",
        )

        from src.retrieval.colbert.indexer import build_index, retrieve

        build_index("test_category", [doc_file])
        result = retrieve(
            "test_category", ["agreement", "compensation", "termination"], top_k=3
        )

        assert len(result) <= 3


class TestGetRetrievedChunks:
    def test_formats_retrieved_chunks(self, tmp_category_dir, tmp_path):
        doc_file = tmp_path / "doc1.txt"
        doc_file.write_text(
            "The agreement number is SA-001. Compensation is hourly.", encoding="utf-8"
        )

        from src.retrieval.colbert.indexer import build_index
        from src.retrieval.colbert.retriever import get_retrieved_chunks

        build_index("test_category", [doc_file])
        result = get_retrieved_chunks("test_category", ["agreement number"], top_k=5)

        assert len(result) >= 1
        assert "content" in result[0]
        assert "score" in result[0]
        assert "rank" in result[0]
        assert result[0]["rank"] == 1
