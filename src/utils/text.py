import re
from pathlib import Path


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def truncate_to_tokens(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _extract_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def _extract_rtf(path: Path) -> str:
    from striprtf.striprtf import rtf_to_text

    raw = path.read_text(encoding="utf-8")
    return rtf_to_text(raw)


def _extract_odt(path: Path) -> str:
    from odf.opendocument import load
    from odf.text import P

    doc = load(str(path))
    paragraphs = doc.getElementsByType(P)
    return "\n".join(str(p) for p in paragraphs if str(p).strip())


def _extract_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8")


_TEXT_EXTRACTORS: dict[str, callable] = {
    ".docx": _extract_docx,
    ".html": _extract_html,
    ".htm": _extract_html,
    ".rtf": _extract_rtf,
    ".odt": _extract_odt,
}

_PLAIN_TEXT_EXTENSIONS: set[str] = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".log",
    ".tsv",
}


def extract_text_from_file(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    suffix = path.suffix.lower()

    extractor = _TEXT_EXTRACTORS.get(suffix)
    if extractor:
        try:
            return extractor(path)
        except Exception as e:
            raise ValueError(f"Failed to extract text from {path}: {e}") from e

    if suffix in _PLAIN_TEXT_EXTENSIONS or suffix not in _TEXT_EXTRACTORS:
        try:
            return _extract_plain(path)
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Cannot read {path} as text (binary or unsupported encoding): {e}"
            ) from e
