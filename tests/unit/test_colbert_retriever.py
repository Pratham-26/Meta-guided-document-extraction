from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest


class TestBuildIndex:
    def test_creates_index_from_documents(self, tmp_category_dir, tmp_path):
        mock_model = MagicMock()
        mock_rag_cls = MagicMock()
        mock_rag_cls.from_pretrained.return_value = mock_model

        doc_file = tmp_path / "doc1.txt"
        doc_file.write_text("Sample document content.", encoding="utf-8")

        with patch.dict(
            "sys.modules",
            {"ragatouille": MagicMock(RAGPretrainedModel=mock_rag_cls)},
            clear=False,
        ):
            from src.retrieval.colbert.indexer import build_index

            build_index("test_category", [doc_file])

        mock_model.index.assert_called_once()
        call_kwargs = mock_model.index.call_args
        assert call_kwargs.kwargs["index_name"] == "test_category"
        assert len(call_kwargs.kwargs["collection"]) == 1

    def test_handles_string_documents(self, tmp_category_dir):
        mock_model = MagicMock()
        mock_rag_cls = MagicMock()
        mock_rag_cls.from_pretrained.return_value = mock_model

        with patch.dict(
            "sys.modules",
            {"ragatouille": MagicMock(RAGPretrainedModel=mock_rag_cls)},
            clear=False,
        ):
            from src.retrieval.colbert.indexer import build_index

            build_index("test_category", ["Plain text document content"])

        call_kwargs = mock_model.index.call_args
        assert call_kwargs.kwargs["collection"] == ["Plain text document content"]


class TestRetrieve:
    def test_retrieves_and_deduplicates(self, tmp_category_dir):
        mock_model = MagicMock()
        mock_model.search.return_value = [
            {"content": "chunk A", "score": 0.9},
            {"content": "chunk B", "score": 0.8},
        ]
        mock_rag_cls = MagicMock()
        mock_rag_cls.from_index.return_value = mock_model

        with patch.dict(
            "sys.modules",
            {"ragatouille": MagicMock(RAGPretrainedModel=mock_rag_cls)},
            clear=False,
        ):
            from src.retrieval.colbert.indexer import retrieve

            result = retrieve("test_category", ["query"], top_k=5)

        assert len(result) == 2
        assert result[0]["content"] == "chunk A"
        assert result[1]["content"] == "chunk B"

    def test_deduplicates_duplicate_content(self, tmp_category_dir):
        mock_model = MagicMock()
        mock_model.search.return_value = [
            {"content": "same chunk", "score": 0.95},
            {"content": "same chunk", "score": 0.85},
            {"content": "unique chunk", "score": 0.7},
        ]
        mock_rag_cls = MagicMock()
        mock_rag_cls.from_index.return_value = mock_model

        with patch.dict(
            "sys.modules",
            {"ragatouille": MagicMock(RAGPretrainedModel=mock_rag_cls)},
            clear=False,
        ):
            from src.retrieval.colbert.indexer import retrieve

            result = retrieve("test_category", ["query"], top_k=5)

        assert len(result) == 2

    def test_respects_top_k(self, tmp_category_dir):
        mock_model = MagicMock()
        mock_model.search.return_value = [
            {"content": f"chunk {i}", "score": 0.9 - i * 0.1} for i in range(10)
        ]
        mock_rag_cls = MagicMock()
        mock_rag_cls.from_index.return_value = mock_model

        with patch.dict(
            "sys.modules",
            {"ragatouille": MagicMock(RAGPretrainedModel=mock_rag_cls)},
            clear=False,
        ):
            from src.retrieval.colbert.indexer import retrieve

            result = retrieve("test_category", ["query"], top_k=3)

        assert len(result) == 3


class TestGetRetrievedChunks:
    def test_formats_retrieved_chunks(self, tmp_category_dir):
        raw_results = [
            {"content": "chunk A", "score": 0.9},
            {"content": "chunk B", "score": 0.8},
        ]

        with patch("src.retrieval.colbert.indexer.retrieve", return_value=raw_results):
            from src.retrieval.colbert.retriever import get_retrieved_chunks

            result = get_retrieved_chunks("test_category", ["query"], top_k=5)

        assert len(result) == 2
        assert result[0]["content"] == "chunk A"
        assert result[0]["score"] == 0.9
        assert result[0]["rank"] == 1
        assert result[1]["rank"] == 2

    def test_handles_document_key_fallback(self, tmp_category_dir):
        raw_results = [
            {"document": "chunk A", "score": 0.9},
        ]

        with patch("src.retrieval.colbert.indexer.retrieve", return_value=raw_results):
            from src.retrieval.colbert.retriever import get_retrieved_chunks

            result = get_retrieved_chunks("test_category", ["query"])

        assert result[0]["content"] == "chunk A"
