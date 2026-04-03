from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class GoldStandard(BaseModel):
    id: str
    category: str
    input_modality: str
    source_document_uri: Path
    extraction: dict
    approved_by: str
    created_at: datetime
    supersedes: str | None = None
