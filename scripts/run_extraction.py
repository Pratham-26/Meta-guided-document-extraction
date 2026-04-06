import argparse
import json
from pathlib import Path

from src.orchestration.graph import run_pipeline
from src.orchestration.state import PipelineState
from src.schemas.document import DocumentInput, InputType


def main():
    parser = argparse.ArgumentParser(description="Run extraction on a document")
    parser.add_argument("--document", required=True, help="Path to the document")
    parser.add_argument("--category", required=True, help="Category name")
    parser.add_argument(
        "--gold",
        action="store_true",
        help="Flag this document as a gold standard candidate",
    )
    args = parser.parse_args()

    doc_path = Path(args.document)
    if not doc_path.exists():
        print(f"Document not found: {doc_path}")
        return

    input_type = InputType.PDF if doc_path.suffix.lower() == ".pdf" else InputType.TEXT

    document = DocumentInput(
        source_uri=doc_path,
        input_type=input_type,
        category=args.category,
    )

    state: PipelineState = {
        "document": document,
        "category_name": args.category,
        "input_modality": "pdf" if input_type == InputType.PDF else "text",
        "is_gold_doc": args.gold,
        "gold_source": "user_flag" if args.gold else None,
    }

    result = run_pipeline(state)

    if result.get("error"):
        print(f"Error: {result['error']}")
        return

    extraction = result.get("extraction", {})
    print(json.dumps(extraction, indent=2))

    evaluation = result.get("judge_evaluation")
    if evaluation:
        print(
            f"\nJudge: {evaluation.quality_tier.value} (confidence: {evaluation.confidence:.2f})"
        )
        if evaluation.feedback:
            print(f"Feedback: {evaluation.feedback}")

    if result.get("is_gold_doc"):
        print(
            f"\n[gold] Document processed as gold standard ({result.get('gold_source', 'unknown')})"
        )


if __name__ == "__main__":
    main()
