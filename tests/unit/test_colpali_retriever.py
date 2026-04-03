from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest


class TestBuildIndex:
    def test_build_index_creates_index_dir(self, tmp_category_dir):
        fake_image = MagicMock()

        mock_pdf2image = MagicMock()
        mock_pdf2image.convert_from_path.return_value = [fake_image]

        mock_colpali_models = MagicMock()
        mock_colpali_models.ColPali.from_pretrained.return_value = MagicMock()
        mock_proc = MagicMock()
        mock_proc.process_images.return_value = {"pixel_values": []}
        mock_colpali_models.ColPaliProcessor.from_pretrained.return_value = mock_proc

        mock_colpali_engine = MagicMock()
        mock_colpali_engine.models = mock_colpali_models

        mock_pickle_mod = MagicMock()
        mock_pickle_file = MagicMock()
        mock_pickle_file.__enter__ = MagicMock(return_value=mock_pickle_file)
        mock_pickle_file.__exit__ = MagicMock(return_value=False)
        mock_pickle_mod.open.return_value = mock_pickle_file

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

            result = indexer_mod.build_index("test_category", Path("test.pdf"))

        assert result.exists()
        assert result.name == "colpali"
        mock_pickle_mod.dump.assert_called_once()
        dumped_data = mock_pickle_mod.dump.call_args[0][0]
        assert dumped_data["model_name"] == "vidore/colpali-v1.2"
        assert dumped_data["num_pages"] == 1
        assert dumped_data["source_pdf"] == str(Path("test.pdf"))


class TestRetrieve:
    def _create_index_pkl(self, category, num_pages):
        import pickle
        from src.storage.paths import colpali_index_dir

        index_dir = colpali_index_dir(category)
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "index.pkl", "wb") as f:
            pickle.dump(
                {
                    "model_name": "vidore/colpali-v1.2",
                    "num_pages": num_pages,
                    "page_embeddings": [None] * num_pages,
                    "source_pdf": "test.pdf",
                },
                f,
            )

    def _mock_heavy_deps(self, score=0.9):
        mock_colpali_models = MagicMock()
        mock_colpali_models.ColPali.from_pretrained.return_value = MagicMock()
        mock_colpali_models.ColPaliProcessor.from_pretrained.return_value = MagicMock(
            process_queries=MagicMock(return_value={"input_ids": []})
        )

        mock_colpali = MagicMock()
        mock_colpali.models = mock_colpali_models

        mock_torch = MagicMock()
        score_result = MagicMock()
        score_result.item.return_value = score
        mock_torch.einsum.return_value.max.return_value.values.sum.return_value = (
            score_result
        )

        return {
            "colpali_engine": mock_colpali,
            "colpali_engine.models": mock_colpali_models,
            "torch": mock_torch,
        }

    def test_retrieve_returns_page_indices(self, tmp_category_dir):
        self._create_index_pkl("test_category", 3)
        modules = self._mock_heavy_deps(score=0.9)

        with patch.dict("sys.modules", modules, clear=False):
            from src.retrieval.colpali.indexer import retrieve

            result = retrieve("test_category", ["Who is the tenant?"], top_k=2)

        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)
        assert len(result) <= 2

    def test_retrieve_deduplicates_pages(self, tmp_category_dir):
        self._create_index_pkl("test_category", 2)
        modules = self._mock_heavy_deps(score=0.9)

        with patch.dict("sys.modules", modules, clear=False):
            from src.retrieval.colpali.indexer import retrieve

            result = retrieve("test_category", ["query1", "query2"], top_k=5)

        assert len(result) == len(set(result))


class TestGetRetrievedPages:
    def _create_index_pkl(self, category, source_pdf="test.pdf"):
        import pickle
        from src.storage.paths import colpali_index_dir

        index_dir = colpali_index_dir(category)
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "index.pkl", "wb") as f:
            pickle.dump(
                {
                    "model_name": "vidore/colpali-v1.2",
                    "source_pdf": source_pdf,
                    "page_embeddings": [],
                    "num_pages": 5,
                },
                f,
            )

    def test_get_retrieved_pages_returns_page_dicts(self, tmp_category_dir):
        self._create_index_pkl("test_category")

        mock_pdf2image = MagicMock()
        mock_pdf2image.convert_from_path.return_value = [MagicMock()] * 5

        with patch("src.retrieval.colpali.indexer.retrieve", return_value=[0, 2]):
            with patch.dict("sys.modules", {"pdf2image": mock_pdf2image}, clear=False):
                from src.retrieval.colpali.retriever import get_retrieved_pages

                result = get_retrieved_pages("test_category", ["query"], top_k=2)

        assert len(result) == 2
        assert result[0]["page_number"] == 1
        assert result[1]["page_number"] == 3

    def test_get_retrieved_pages_skips_out_of_range_indices(self, tmp_category_dir):
        self._create_index_pkl("test_category")

        mock_pdf2image = MagicMock()
        mock_pdf2image.convert_from_path.return_value = [MagicMock()] * 3

        with patch("src.retrieval.colpali.indexer.retrieve", return_value=[99]):
            with patch.dict("sys.modules", {"pdf2image": mock_pdf2image}, clear=False):
                from src.retrieval.colpali.retriever import get_retrieved_pages

                result = get_retrieved_pages("test_category", ["query"], top_k=3)

        assert len(result) == 0
