import argparse
import json
from pathlib import Path

from src.config.lm import get_lm
from src.config.loader import load_category_config
from src.config.settings import settings
from src.orchestration.graph import compile_graph
from src.orchestration.state import PipelineState
from src.schemas.document import DocumentInput, InputType


def main():
    parser = argparse.ArgumentParser(description="Run extraction on a document")
    parser.add_argument("--document", required=True, help="Path to the document")
    parser.add_argument("--category", required=True, help="Category name")
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
    }

    app = compile_graph()
    result = app.invoke(state)

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


if __name__ == "__main__":
    main()
