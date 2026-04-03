from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class InputType(str, Enum):
    PDF = "pdf"
    TEXT = "text"


class DocumentInput(BaseModel):
    source_uri: Path
    input_type: InputType
    category: str
    raw_text: str | None = None
    metadata: dict = {}
