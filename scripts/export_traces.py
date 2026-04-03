import argparse
import json
from pathlib import Path

from src.storage.trace_logger import read_traces
from src.config.settings import settings


def main():
    parser = argparse.ArgumentParser(description="Export trace logs for SLM training")
    parser.add_argument("--category", required=True, help="Category name")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--phase",
        default=None,
        help="Filter by phase (extraction, evaluation, optimization)",
    )
    parser.add_argument("--date", default=None, help="Filter by date (YYYY-MM-DD)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    phases = (
        [args.phase] if args.phase else ["extraction", "evaluation", "optimization"]
    )
    total = 0

    for phase in phases:
        traces = read_traces(args.category, phase, args.date)
        if not traces:
            print(f"No traces found for phase '{phase}'.")
            continue

        out_file = output_dir / f"{args.category}_{phase}_traces.jsonl"
        with open(out_file, "w") as f:
            for t in traces:
                f.write(t.model_dump_json() + "\n")

        print(f"Exported {len(traces)} {phase} traces to {out_file}")
        total += len(traces)

    print(f"\nTotal: {total} trace entries exported.")


if __name__ == "__main__":
    main()
