from datetime import datetime, timezone

from src.schemas.question import QuestionEntry, QuestionSet
from src.storage.file_lock import locked_file
from src.storage.fs_store import load_question_set, save_question_set
from src.storage import paths


def get_questions(category: str, modality: str) -> list[str]:
    qs = load_question_set(category, modality)
    if not qs:
        return []
    return [q.text for q in qs.questions]


def add_questions(
    category: str, modality: str, new_questions: list[dict]
) -> QuestionSet:
    q_path = paths.questions_path(category, modality)

    with locked_file(q_path):
        existing = load_question_set(category, modality)
        existing_ids = {q.id for q in existing.questions} if existing else set()
        next_idx = len(existing.questions) if existing else 0

        entries = list(existing.questions) if existing else []
        for i, q in enumerate(new_questions):
            qid = f"q_{next_idx + i + 1:03d}"
            if qid not in existing_ids:
                entries.append(
                    QuestionEntry(
                        id=qid,
                        text=q["text"],
                        target_field=q.get("target_field", "unknown"),
                        retrieval_priority=q.get("retrieval_priority", 1),
                    )
                )

        qs = QuestionSet(
            category=category,
            input_modality=modality,
            version=(existing.version + 1) if existing else 1,
            updated_at=datetime.now(timezone.utc).isoformat(),
            questions=entries,
        )
        save_question_set(category, modality, qs)
    return qs


def merge_questions(
    category: str, modality: str, new_questions: list[dict]
) -> QuestionSet:
    q_path = paths.questions_path(category, modality)

    with locked_file(q_path):
        existing = load_question_set(category, modality)

        if not existing:
            return add_questions(category, modality, new_questions)

        existing_fields = {q.target_field for q in existing.questions}
        entries = list(existing.questions)

        for q in new_questions:
            target_field = q.get("target_field", "unknown")
            if target_field not in existing_fields:
                next_idx = len(entries) + 1
                entries.append(
                    QuestionEntry(
                        id=f"q_{next_idx:03d}",
                        text=q["text"],
                        target_field=target_field,
                        retrieval_priority=q.get("retrieval_priority", 1),
                    )
                )
                existing_fields.add(target_field)

        qs = QuestionSet(
            category=category,
            input_modality=modality,
            version=existing.version + 1,
            updated_at=datetime.now(timezone.utc).isoformat(),
            questions=entries,
        )
        save_question_set(category, modality, qs)
    return qs
