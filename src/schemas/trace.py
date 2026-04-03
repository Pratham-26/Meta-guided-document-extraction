from datetime import datetime

from pydantic import BaseModel


class TraceEntry(BaseModel):
    timestamp: datetime
    agent_role: str
    phase: str
    category: str
    prompt: str
    response: str
    model: str
    provider: str
    token_usage: dict
    quality_tier: str | None = None
    document_id: str | None = None
