from pathlib import Path

from src.retrieval.router import route, RetrievalRoute
from src.schemas.document import InputType


class TestRouter:
    def test_pdf_routes_to_colpali(self):
        result = route(InputType.PDF, Path("test.pdf"))
        assert result == RetrievalRoute.COLPALI

    def test_text_routes_to_colbert(self):
        result = route(InputType.TEXT, Path("test.txt"))
        assert result == RetrievalRoute.COLBERT

    def test_pdf_without_path(self):
        result = route(InputType.PDF)
        assert result == RetrievalRoute.COLPALI

    def test_text_without_path(self):
        result = route(InputType.TEXT)
        assert result == RetrievalRoute.COLBERT
