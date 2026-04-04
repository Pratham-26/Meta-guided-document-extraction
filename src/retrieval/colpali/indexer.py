from pathlib import Path

from src.storage import paths


def build_index(
    category: str,
    pdf_paths: list[Path],
    index_dir: Path | None = None,
):
    from colpali_engine.models import ColPali, ColPaliProcessor
    from pdf2image import convert_from_path

    out_dir = index_dir or paths.colpali_index_dir(category)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = "vidore/colpali-v1.2"
    model = ColPali.from_pretrained(model_name)
    processor = ColPaliProcessor.from_pretrained(model_name)

    page_embeddings = []
    page_sources = []

    for pdf_path in pdf_paths:
        images = convert_from_path(str(pdf_path))
        for page_idx, img in enumerate(images):
            proc = processor.process_images([img])
            emb = model(**proc)
            page_embeddings.append(emb)
            page_sources.append({"source_pdf": str(pdf_path), "page_in_pdf": page_idx})

    import pickle

    data = {
        "model_name": model_name,
        "num_pages": len(page_embeddings),
        "page_embeddings": [e.cpu().detach() for e in page_embeddings],
        "page_sources": page_sources,
    }
    with open(out_dir / "index.pkl", "wb") as f:
        pickle.dump(data, f)

    return out_dir


def retrieve(
    category: str,
    queries: list[str],
    top_k: int = 3,
    index_dir: Path | None = None,
) -> tuple[list[int], list[dict]]:
    import pickle
    import torch

    from colpali_engine.models import ColPali, ColPaliProcessor

    search_dir = index_dir or paths.colpali_index_dir(category)
    with open(search_dir / "index.pkl", "rb") as f:
        data = pickle.load(f)

    model_name = data["model_name"]
    model = ColPali.from_pretrained(model_name)
    processor = ColPaliProcessor.from_pretrained(model_name)

    page_embeddings = data["page_embeddings"]
    scores = []

    for query in queries:
        q_proc = processor.process_queries([query])
        q_emb = model(**q_proc).cpu()

        for page_idx, p_emb in enumerate(page_embeddings):
            score = (
                torch.einsum("bnd,csd->bnsc", q_emb, p_emb)
                .max(dim=2)
                .values.sum()
                .item()
            )
            scores.append((page_idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    top_pages = []
    for page_idx, score in scores:
        if page_idx not in seen:
            seen.add(page_idx)
            top_pages.append(page_idx)
            if len(top_pages) >= top_k:
                break

    return top_pages, data["page_sources"]


def rebuild_from_gold_sources(category: str):
    from src.storage.fs_store import list_gold_standards

    gold_standards = list_gold_standards(category, "pdf")
    pdf_paths = []
    for gs in gold_standards:
        p = Path(gs.source_document_uri)
        if p.exists() and p not in pdf_paths:
            pdf_paths.append(p)

    if not pdf_paths:
        return

    build_index(category, pdf_paths)
