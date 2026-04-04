from pathlib import Path

from src.storage import paths


def get_retrieved_pages(
    category: str,
    queries: list[str],
    top_k: int = 3,
    index_dir: Path | None = None,
) -> list[dict]:
    from pdf2image import convert_from_path

    from src.retrieval.colpali.indexer import retrieve

    page_indices, page_sources = retrieve(category, queries, top_k, index_dir=index_dir)

    _image_cache: dict[str, list] = {}
    results = []
    for idx in page_indices:
        if idx >= len(page_sources):
            continue
        src = page_sources[idx]
        pdf_path = src["source_pdf"]
        page_in_pdf = src["page_in_pdf"]

        if pdf_path not in _image_cache:
            _image_cache[pdf_path] = convert_from_path(pdf_path)
        images = _image_cache[pdf_path]

        if page_in_pdf < len(images):
            results.append(
                {
                    "page_number": page_in_pdf + 1,
                    "image": images[page_in_pdf],
                    "source_pdf": pdf_path,
                }
            )

    return results
