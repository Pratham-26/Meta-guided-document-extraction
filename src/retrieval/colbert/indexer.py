from pathlib import Path

from src.storage import paths


def build_index(
    category: str, documents: list[str] | list[Path], index_name: str | None = None
):
    from ragatouille import RAGPretrainedModel

    index_dir = paths.colbert_index_dir(category)
    index_dir.mkdir(parents=True, exist_ok=True)

    model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    name = index_name or category
    doc_texts = []
    for doc in documents:
        p = Path(doc) if not isinstance(doc, Path) else doc
        if p.exists():
            doc_texts.append(p.read_text(encoding="utf-8"))
        else:
            doc_texts.append(str(doc))

    model.index(
        collection=doc_texts,
        index_name=name,
        max_document_length=180,
        split_documents=True,
    )

    return model


def retrieve(category: str, queries: list[str], top_k: int = 5) -> list[dict]:
    from ragatouille import RAGPretrainedModel

    index_dir = paths.colbert_index_dir(category)
    model = RAGPretrainedModel.from_index(str(index_dir))

    all_results = []
    for query in queries:
        results = model.search(query, k=top_k)
        all_results.extend(results)

    seen = set()
    deduped = []
    for r in sorted(all_results, key=lambda x: x.get("score", 0), reverse=True):
        content = r.get("content", r.get("document", ""))
        if content not in seen:
            seen.add(content)
            deduped.append(r)

    return deduped[:top_k]
