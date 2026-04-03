from pathlib import Path


def get_retrieved_pages(
    category: str, queries: list[str], top_k: int = 3
) -> list[dict]:
    from src.retrieval.colpali.indexer import retrieve
    from pdf2image import convert_from_path
    from src.storage import paths

    page_indices = retrieve(category, queries, top_k)

    index_dir = paths.colpali_index_dir(category)
    import pickle

    with open(index_dir / "index.pkl", "rb") as f:
        data = pickle.load(f)

    source_pdf = Path(data["source_pdf"])
    images = convert_from_path(str(source_pdf))

    results = []
    for idx in page_indices:
        if idx < len(images):
            results.append(
                {
                    "page_number": idx + 1,
                    "image": images[idx],
                    "source_pdf": str(source_pdf),
                }
            )

    return results
