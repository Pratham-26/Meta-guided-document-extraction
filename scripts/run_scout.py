import argparse

from src.config.lm import get_lm
from src.config.loader import load_category_config
from src.agents.scout.agent import ScoutAgent
from src.agents.scout.question_store import add_questions
from src.agents.scout.gold_builder import build_and_save
from src.schemas.gold_standard import ApprovalStatus
from src.storage.fs_store import save_source_document, list_gold_standards
from src.utils.pdf import extract_text_from_pdf, load_pdf_pages
from src.utils.text import clean_text, truncate_to_tokens, extract_text_from_file
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run Scout Agent for a category")
    parser.add_argument("--category", required=True, help="Category name")
    parser.add_argument(
        "--doc",
        action="append",
        help="Additional documents to explore (can be repeated)",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve Scout-generated Gold Standards (skip HITL review)",
    )
    args = parser.parse_args()

    config = load_category_config(args.category)

    approval = (
        ApprovalStatus.APPROVED if args.auto_approve else ApprovalStatus.PENDING_REVIEW
    )

    documents = config.sample_documents[:]
    if args.doc:
        documents.extend(args.doc)

    explorations = []
    scout: ScoutAgent | None = None

    for i, doc_path in enumerate(documents):
        path = Path(doc_path)
        print(f"Exploring document {i + 1}/{len(documents)}: {path.name}")

        is_pdf = path.suffix.lower() == ".pdf"
        modality = "pdf" if is_pdf else "text"

        saved = save_source_document(args.category, modality, path)

        images = None
        if is_pdf:
            content = extract_text_from_pdf(path)
            content = clean_text(content)
            content = truncate_to_tokens(content)
            is_placeholder = content.startswith("[PDF with")

            if is_placeholder:
                print(
                    "  Warning: PyMuPDF not available, loading page images for vision exploration"
                )
                images = load_pdf_pages(path)

            lm = get_lm("scout", input_type="vision")
            vision_lm = get_lm("scout", input_type="vision")
            if scout is None:
                scout = ScoutAgent(lm=lm, vision_lm=vision_lm)
        else:
            content = extract_text_from_file(path)
            content = clean_text(content)
            content = truncate_to_tokens(content)

            lm = get_lm("scout", input_type="text")
            if scout is None:
                scout = ScoutAgent(lm=lm)

        result = scout.explore_document(
            content=content,
            schema=config.expected_schema,
            instructions=config.extraction_instructions,
            images=images,
        )
        explorations.append(result["exploration"])

        gs_id = f"gs_{i + 1:03d}"
        build_and_save(
            category=args.category,
            modality=modality,
            gs_id=gs_id,
            source_document_uri=saved,
            extraction=result["extraction"],
            approval_status=approval,
        )
        status_label = "approved" if args.auto_approve else "pending review"
        print(f"  Gold Standard saved: {gs_id} ({status_label})")

    if scout is None:
        print("No documents to explore.")
        return

    print("\nInferring questions from explorations...")
    questions = scout.infer_questions_from_explorations(
        explorations=explorations,
        schema=config.expected_schema,
        instructions=config.extraction_instructions,
    )

    if questions:
        qs = add_questions(args.category, modality, questions)
        print(f"Inferred {len(qs.questions)} questions:")
        for q in qs.questions:
            print(f"  [{q.id}] {q.text} -> {q.target_field}")

    gs_count = len(list_gold_standards(args.category, modality))
    print(f"\nScout complete. {gs_count} Gold Standards, {len(questions)} questions.")

    if not args.auto_approve:
        print(
            "\n⚠  Gold Standards are PENDING REVIEW. Approve them via the HITL UI:"
        )
        print("  python scripts/review_server.py")

    if modality == "pdf":
        from src.retrieval.colpali.indexer import rebuild_from_gold_sources

        rebuild_from_gold_sources(args.category)
    else:
        from src.retrieval.colbert.indexer import rebuild_from_gold_sources

        rebuild_from_gold_sources(args.category)
    print("Index rebuilt from gold sources.")


if __name__ == "__main__":
    main()
