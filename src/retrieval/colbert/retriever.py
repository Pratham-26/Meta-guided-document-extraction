def get_retrieved_chunks(
    category: str, queries: list[str], top_k: int = 5
) -> list[dict]:
    from src.retrieval.colbert.indexer import retrieve

    results = retrieve(category, queries, top_k)
    return [
        {
            "content": r.get("content", r.get("document", "")),
            "score": r.get("score", 0),
            "rank": i + 1,
        }
        for i, r in enumerate(results)
    ]
