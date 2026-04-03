from enum import Enum
from pathlib import Path

from src.schemas.document import InputType


class RetrievalRoute(str, Enum):
    COLPALI = "colpali"
    COLBERT = "colbert"


def route(input_type: InputType, file_path: Path | None = None) -> RetrievalRoute:
    if input_type == InputType.PDF:
        return RetrievalRoute.COLPALI
    return RetrievalRoute.COLBERT
