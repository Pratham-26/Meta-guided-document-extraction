from pathlib import Path

from src.storage import paths
from src.storage.fs_store import list_gold_standards
from src.utils.text import extract_text_from_file


def build_index(
    category: str,
    documents: list[str] | list[Path],
    index_name: str | None = None,
    index_dir: Path | None = None,
):
    from ragatouille import RAGPretrainedModel

    out_dir = index_dir or paths.colbert_index_dir(category)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    name = index_name or category
    doc_texts = []
    for doc in documents:
        p = Path(doc) if not isinstance(doc, Path) else doc
        if p.exists():
            doc_texts.append(extract_text_from_file(p))
        else:
            doc_texts.append(str(doc))

    model.index(
        collection=doc_texts,
        index_name=name,
        index_folder=str(out_dir),
        max_document_length=180,
        split_documents=True,
    )

    return model


def retrieve(
    category: str,
    queries: list[str],
    top_k: int = 5,
    index_dir: Path | None = None,
) -> list[dict]:
    from ragatouille import RAGPretrainedModel

    search_dir = index_dir or paths.colbert_index_dir(category)
    model = RAGPretrainedModel.from_index(str(search_dir))

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


def rebuild_from_gold_sources(category: str):
    gold_standards = list_gold_standards(category, "text")
    source_paths = []
    for gs in gold_standards:
        p = Path(gs.source_document_uri)
        if p.exists():
            source_paths.append(p)

    if not source_paths:
        return

    build_index(category, source_paths)
