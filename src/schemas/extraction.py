from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class ExtractionResult(BaseModel):
    document_id: str
    category: str
    source_uri: Path
    extraction: dict
    extracted_at: datetime
    prompt_version: str | None = None
    model: str | None = None
    retrieval_context: dict = {}
