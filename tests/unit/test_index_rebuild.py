import pickle
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.schemas.gold_standard import GoldStandard
from src.storage.fs_store import save_gold_standard
from src.storage.paths import ensure_category_dirs, colpali_index_dir


def _make_gs(category, modality, gs_id, source_uri):
    return GoldStandard(
        id=gs_id,
        category=category,
        input_modality=modality,
        source_document_uri=Path(source_uri),
        extraction={"name": "Test", "amount": 100},
        approved_by="scout",
        created_at=datetime.now(timezone.utc),
    )


class TestColbertRebuildFromGoldSources:
    def test_rebuild_calls_build_index_with_gold_source_paths(
        self, tmp_category_dir, tmp_path
    ):
        ensure_category_dirs("test_category")

        doc1 = tmp_path / "doc1.txt"
        doc1.write_text("Content from gold doc 1", encoding="utf-8")
        doc2 = tmp_path / "doc2.txt"
        doc2.write_text("Content from gold doc 2", encoding="utf-8")

        save_gold_standard(
            "test_category",
            "text",
            _make_gs("test_category", "text", "gs_001", str(doc1)),
        )
        save_gold_standard(
            "test_category",
            "text",
            _make_gs("test_category", "text", "gs_002", str(doc2)),
        )

        mock_model = MagicMock()
        mock_rag_cls = MagicMock()
        mock_rag_cls.from_pretrained.return_value = mock_model

        with patch.dict(
            "sys.modules",
            {"ragatouille": MagicMock(RAGPretrainedModel=mock_rag_cls)},
            clear=False,
        ):
            from src.retrieval.colbert.indexer import rebuild_from_gold_sources

            rebuild_from_gold_sources("test_category")

        mock_model.index.assert_called_once()
        collection = mock_model.index.call_args.kwargs["collection"]
        assert len(collection) == 2

    def test_rebuild_skips_nonexistent_sources(self, tmp_category_dir):
        ensure_category_dirs("test_category")

        save_gold_standard(
            "test_category",
            "text",
            _make_gs("test_category", "text", "gs_001", "/nonexistent/file.txt"),
        )

        mock_model = MagicMock()
        mock_rag_cls = MagicMock()
        mock_rag_cls.from_pretrained.return_value = mock_model

        with patch.dict(
            "sys.modules",
            {"ragatouille": MagicMock(RAGPretrainedModel=mock_rag_cls)},
            clear=False,
        ):
            from src.retrieval.colbert.indexer import rebuild_from_gold_sources

            rebuild_from_gold_sources("test_category")

        mock_model.index.assert_not_called()

    def test_rebuild_with_no_gold_standards(self, tmp_category_dir):
        ensure_category_dirs("test_category")

        with patch(
            "src.retrieval.colbert.indexer.list_gold_standards", return_value=[]
        ):
            from src.retrieval.colbert.indexer import rebuild_from_gold_sources

            result = rebuild_from_gold_sources("test_category")

        assert result is None


class TestColpaliRebuildFromGoldSources:
    def test_rebuild_builds_index_from_gold_pdfs(self, tmp_category_dir, tmp_path):
        ensure_category_dirs("test_category")

        pdf1 = tmp_path / "gold1.pdf"
        pdf1.write_bytes(b"fake pdf 1")
        pdf2 = tmp_path / "gold2.pdf"
        pdf2.write_bytes(b"fake pdf 2")

        save_gold_standard(
            "test_category",
            "pdf",
            _make_gs("test_category", "pdf", "gs_001", str(pdf1)),
        )
        save_gold_standard(
            "test_category",
            "pdf",
            _make_gs("test_category", "pdf", "gs_002", str(pdf2)),
        )

        mock_pdf2image = MagicMock()
        mock_pdf2image.convert_from_path.return_value = [MagicMock()]

        mock_colpali_models = MagicMock()
        mock_colpali_models.ColPali.from_pretrained.return_value = MagicMock()
        mock_proc = MagicMock()
        mock_proc.process_images.return_value = {"pixel_values": []}
        mock_colpali_models.ColPaliProcessor.from_pretrained.return_value = mock_proc

        mock_colpali_engine = MagicMock()
        mock_colpali_engine.models = mock_colpali_models

        mock_pickle_mod = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "pdf2image": mock_pdf2image,
                "colpali_engine": mock_colpali_engine,
                "colpali_engine.models": mock_colpali_models,
                "torch": MagicMock(),
                "pickle": mock_pickle_mod,
            },
            clear=False,
        ):
            import importlib
            import src.retrieval.colpali.indexer as indexer_mod

            importlib.reload(indexer_mod)
            indexer_mod.rebuild_from_gold_sources("test_category")

        mock_pickle_mod.dump.assert_called_once()
        dumped_data = mock_pickle_mod.dump.call_args[0][0]
        assert len(dumped_data["page_sources"]) == 2
        assert dumped_data["page_sources"][0]["source_pdf"] == str(pdf1)
        assert dumped_data["page_sources"][1]["source_pdf"] == str(pdf2)

    def test_rebuild_deduplicates_duplicate_source_pdfs(
        self, tmp_category_dir, tmp_path
    ):
        ensure_category_dirs("test_category")

        pdf1 = tmp_path / "gold1.pdf"
        pdf1.write_bytes(b"fake pdf")

        save_gold_standard(
            "test_category",
            "pdf",
            _make_gs("test_category", "pdf", "gs_001", str(pdf1)),
        )
        save_gold_standard(
            "test_category",
            "pdf",
            _make_gs("test_category", "pdf", "gs_002", str(pdf1)),
        )

        mock_pdf2image = MagicMock()
        mock_pdf2image.convert_from_path.return_value = [MagicMock()]

        mock_colpali_models = MagicMock()
        mock_colpali_models.ColPali.from_pretrained.return_value = MagicMock()
        mock_proc = MagicMock()
        mock_proc.process_images.return_value = {"pixel_values": []}
        mock_colpali_models.ColPaliProcessor.from_pretrained.return_value = mock_proc

        mock_colpali_engine = MagicMock()
        mock_colpali_engine.models = mock_colpali_models

        mock_pickle_mod = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "pdf2image": mock_pdf2image,
                "colpali_engine": mock_colpali_engine,
                "colpali_engine.models": mock_colpali_models,
                "torch": MagicMock(),
                "pickle": mock_pickle_mod,
            },
            clear=False,
        ):
            import importlib
            import src.retrieval.colpali.indexer as indexer_mod

            importlib.reload(indexer_mod)
            indexer_mod.rebuild_from_gold_sources("test_category")

        dumped_data = mock_pickle_mod.dump.call_args[0][0]
        assert len(dumped_data["page_sources"]) == 1


class TestColpaliMultiPdfRetrieval:
    def test_get_retrieved_pages_with_multiple_pdfs(self, tmp_category_dir):
        ensure_category_dirs("test_category")

        index_dir = colpali_index_dir("test_category")
        index_dir.mkdir(parents=True, exist_ok=True)

        with open(index_dir / "index.pkl", "wb") as f:
            pickle.dump(
                {
                    "model_name": "vidore/colpali-v1.2",
                    "page_sources": [
                        {"source_pdf": "pdf_a.pdf", "page_in_pdf": 0},
                        {"source_pdf": "pdf_a.pdf", "page_in_pdf": 1},
                        {"source_pdf": "pdf_b.pdf", "page_in_pdf": 0},
                    ],
                    "page_embeddings": [],
                    "num_pages": 3,
                },
                f,
            )

        fake_img = MagicMock()

        with patch("src.retrieval.colpali.indexer.retrieve", return_value=[0, 2]):
            with patch(
                "src.retrieval.colpali.retriever.convert_from_path",
                return_value=[fake_img, fake_img],
            ):
                from src.retrieval.colpali.retriever import get_retrieved_pages

                result = get_retrieved_pages("test_category", ["query"], top_k=2)

        assert len(result) == 2
        assert result[0]["source_pdf"] == "pdf_a.pdf"
        assert result[0]["page_number"] == 1
        assert result[1]["source_pdf"] == "pdf_b.pdf"
        assert result[1]["page_number"] == 1
