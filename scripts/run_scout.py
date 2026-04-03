import argparse

from src.config.lm import get_lm
from src.config.loader import load_category_config
from src.agents.scout.agent import ScoutAgent
from src.agents.scout.question_store import add_questions
from src.agents.scout.gold_builder import build_and_save
from src.storage.fs_store import save_source_document, list_gold_standards
from src.utils.pdf import extract_text_from_pdf
from src.utils.text import clean_text, truncate_to_tokens
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run Scout Agent for a category")
    parser.add_argument("--category", required=True, help="Category name")
    parser.add_argument(
        "--doc",
        action="append",
        help="Additional documents to explore (can be repeated)",
    )
    args = parser.parse_args()

    config = load_category_config(args.category)

    documents = config.sample_documents[:]
    if args.doc:
        documents.extend(args.doc)

    explorations = []
    for i, doc_path in enumerate(documents):
        path = Path(doc_path)
        print(f"Exploring document {i + 1}/{len(documents)}: {path.name}")

        saved = save_source_document(args.category, path)

        is_pdf = path.suffix.lower() == ".pdf"
        modality = "pdf" if is_pdf else "text"
        input_type = "vision" if is_pdf else "text"

        if is_pdf:
            content = extract_text_from_pdf(path)
        else:
            content = path.read_text(encoding="utf-8")

        content = clean_text(content)
        content = truncate_to_tokens(content)

        lm = get_lm("scout", input_type=input_type)
        scout = ScoutAgent(lm=lm)

        result = scout.explore_document(
            content=content,
            schema=config.expected_schema,
            instructions=config.extraction_instructions,
        )
        explorations.append(result["exploration"])

        gs_id = f"gs_{i + 1:03d}"
        build_and_save(
            category=args.category,
            modality=modality,
            gs_id=gs_id,
            source_document_uri=saved,
            extraction=result["extraction"],
        )
        print(f"  Gold Standard saved: {gs_id}")

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


if __name__ == "__main__":
    main()
