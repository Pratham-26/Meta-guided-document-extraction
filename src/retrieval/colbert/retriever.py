from pathlib import Path


def get_retrieved_chunks(
    category: str,
    queries: list[str],
    top_k: int = 5,
    index_dir: Path | None = None,
) -> list[dict]:
    from src.retrieval.colbert.indexer import retrieve

    results = retrieve(category, queries, top_k, index_dir=index_dir)
    return [
        {
            "content": r["content"],
            "score": r["score"],
            "rank": i + 1,
        }
        for i, r in enumerate(results)
    ]
