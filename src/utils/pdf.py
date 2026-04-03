from pathlib import Path


def load_pdf_pages(pdf_path: Path) -> list:
    from pdf2image import convert_from_path

    return convert_from_path(str(pdf_path))


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        return "\n".join(page.get_text() for page in doc)
    except ImportError:
        from pdf2image import convert_from_path

        return f"[PDF with {len(convert_from_path(str(pdf_path)))} pages - text extraction requires PyMuPDF]"


def page_to_image(pdf_path: Path, page_num: int = 0):
    from pdf2image import convert_from_path

    images = convert_from_path(
        str(pdf_path), first_page=page_num + 1, last_page=page_num + 1
    )
    return images[0] if images else None
