from src.storage.fs_store import list_gold_standards


def select_examples(category: str, modality: str, max_examples: int = 3) -> list[dict]:
    gold_standards = list_gold_standards(category, modality)
    gold_standards.sort(key=lambda gs: gs.created_at, reverse=True)
    selected = gold_standards[:max_examples]
    return [
        {
            "source": str(gs.source_document_uri),
            "extraction": gs.extraction,
        }
        for gs in selected
    ]
