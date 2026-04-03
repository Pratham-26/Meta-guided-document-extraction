from pydantic import BaseModel


class QuestionEntry(BaseModel):
    id: str
    text: str
    target_field: str
    retrieval_priority: int = 1


class QuestionSet(BaseModel):
    category: str
    input_modality: str
    version: int = 1
    updated_at: str
    questions: list[QuestionEntry]
