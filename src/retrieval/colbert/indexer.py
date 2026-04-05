import json
import math
import re
from collections import Counter
from pathlib import Path

from src.storage import paths
from src.storage.fs_store import list_gold_standards
from src.utils.text import chunk_text, extract_text_from_file


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_tf_idf_index(chunks: list[str]) -> dict:
    N = len(chunks)
    doc_freqs = Counter()
    chunk_tokens = []
    for chunk in chunks:
        tokens = _tokenize(chunk)
        chunk_tokens.append(tokens)
        unique = set(tokens)
        for t in unique:
            doc_freqs[t] += 1

    index = {"chunks": chunks, "idf": {}, "chunk_tokens": chunk_tokens}
    for term, df in doc_freqs.items():
        index["idf"][term] = math.log((N + 1) / (df + 1)) + 1

    return index


def _score_query(query: str, index: dict, top_k: int) -> list[dict]:
    query_tokens = _tokenize(query)
    idf = index["idf"]
    scores = []

    for i, doc_tokens in enumerate(index["chunk_tokens"]):
        token_counts = Counter(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            if qt in token_counts:
                tf = 1 + math.log(token_counts[qt])
                score += tf * idf.get(qt, 1.0)
        scores.append({"content": index["chunks"][i], "score": score})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]


def _save_index(index: dict, index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    data = {"chunks": index["chunks"], "idf": index["idf"]}
    with open(index_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(data, f)


def _load_index(index_dir: Path) -> dict | None:
    path = index_dir / "index.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chunk_tokens = [_tokenize(c) for c in data["chunks"]]
    return {"chunks": data["chunks"], "idf": data["idf"], "chunk_tokens": chunk_tokens}


def build_index(
    category: str,
    documents: list[str] | list[Path],
    index_name: str | None = None,
    index_dir: Path | None = None,
    chunk_size: int = 512,
    overlap: int = 64,
):
    out_dir = index_dir or paths.colbert_index_dir(category)

    all_chunks = []
    for doc in documents:
        p = Path(doc) if not isinstance(doc, Path) else doc
        if p.exists():
            text = extract_text_from_file(p)
        else:
            text = str(doc)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)

    if not all_chunks:
        return None

    index = _build_tf_idf_index(all_chunks)
    _save_index(index, out_dir)
    return index


def retrieve(
    category: str,
    queries: list[str],
    top_k: int = 5,
    index_dir: Path | None = None,
) -> list[dict]:
    search_dir = index_dir or paths.colbert_index_dir(category)
    index = _load_index(search_dir)
    if index is None:
        return []

    all_results = []
    for query in queries:
        results = _score_query(query, index, top_k)
        all_results.extend(results)

    seen = set()
    deduped = []
    for r in sorted(all_results, key=lambda x: x["score"], reverse=True):
        content = r["content"]
        key = content[:200]
        if key not in seen:
            seen.add(key)
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
