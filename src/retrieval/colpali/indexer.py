from pathlib import Path

from src.storage import paths


def build_index(category: str, pdf_path: Path):
    from colpali_engine.models import ColPali, ColPaliProcessor
    from pdf2image import convert_from_path

    index_dir = paths.colpali_index_dir(category)
    index_dir.mkdir(parents=True, exist_ok=True)

    images = convert_from_path(str(pdf_path))

    model_name = "vidore/colpali-v1.2"
    model = ColPali.from_pretrained(model_name)
    processor = ColPaliProcessor.from_pretrained(model_name)

    page_embeddings = []
    for i, img in enumerate(images):
        proc = processor.process_images([img])
        emb = model(**proc)
        page_embeddings.append(emb)

    import torch, pickle

    data = {
        "model_name": model_name,
        "num_pages": len(images),
        "page_embeddings": [e.cpu().detach() for e in page_embeddings],
        "source_pdf": str(pdf_path),
    }
    with open(index_dir / "index.pkl", "wb") as f:
        pickle.dump(data, f)

    return index_dir


def retrieve(category: str, queries: list[str], top_k: int = 3) -> list[int]:
    import pickle
    import torch

    from colpali_engine.models import ColPali, ColPaliProcessor

    index_dir = paths.colpali_index_dir(category)
    with open(index_dir / "index.pkl", "rb") as f:
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

    return top_pages
