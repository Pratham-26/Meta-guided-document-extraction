import json
import sys
import time
from pathlib import Path

from src.config.loader import load_category_config
from src.config.lm import get_lm
from src.agents.extractor.agent import ExtractorAgent
from src.agents.extractor.few_shot import select_examples
from src.utils.text import extract_text_from_file, clean_text, truncate_to_tokens


def deep_compare(expected: dict, actual: dict, path: str = "") -> list[dict]:
    diffs = []

    all_keys = set()
    if isinstance(expected, dict):
        all_keys.update(expected.keys())
    if isinstance(actual, dict):
        all_keys.update(actual.keys())

    for key in all_keys:
        key_path = f"{path}.{key}" if path else key
        exp_val = expected.get(key) if isinstance(expected, dict) else None
        act_val = actual.get(key) if isinstance(actual, dict) else None

        if exp_val is None and act_val is None:
            continue

        if isinstance(exp_val, dict) and isinstance(act_val, dict):
            diffs.extend(deep_compare(exp_val, act_val, key_path))
        elif isinstance(exp_val, list) and isinstance(act_val, list):
            if exp_val and isinstance(exp_val[0], dict):
                max_len = max(len(exp_val), len(act_val))
                for i in range(max_len):
                    item_path = f"{key_path}[{i}]"
                    if i >= len(exp_val):
                        diffs.append(
                            {
                                "path": item_path,
                                "expected": None,
                                "actual": act_val[i],
                                "type": "extra_item",
                            }
                        )
                    elif i >= len(act_val):
                        diffs.append(
                            {
                                "path": item_path,
                                "expected": exp_val[i],
                                "actual": None,
                                "type": "missing_item",
                            }
                        )
                    else:
                        diffs.extend(deep_compare(exp_val[i], act_val[i], item_path))
            elif exp_val != act_val:
                diffs.append(
                    {
                        "path": key_path,
                        "expected": exp_val,
                        "actual": act_val,
                        "type": "value_mismatch",
                    }
                )
        elif exp_val != act_val:
            if exp_val is None and act_val is not None:
                diffs.append(
                    {
                        "path": key_path,
                        "expected": None,
                        "actual": act_val,
                        "type": "extra_field",
                    }
                )
            elif exp_val is not None and act_val is None:
                diffs.append(
                    {
                        "path": key_path,
                        "expected": exp_val,
                        "actual": None,
                        "type": "missing_field",
                    }
                )
            else:
                diffs.append(
                    {
                        "path": key_path,
                        "expected": exp_val,
                        "actual": act_val,
                        "type": "value_mismatch",
                    }
                )

    return diffs


def main():
    category = "service_agreements"
    inputs_dir = Path("examples/service_agreements/inputs")
    expected_dir = Path("examples/service_agreements/expected")

    config = load_category_config(category)
    lm = get_lm("extractor", input_type="text")
    agent = ExtractorAgent(lm=lm)

    examples = select_examples(category, "text")

    input_files = sorted(inputs_dir.glob("sa_*.txt"))

    results = []
    total_diffs = 0
    total_fields_checked = 0
    total_fields_matched = 0

    for i, input_file in enumerate(input_files):
        stem = input_file.stem
        expected_file = expected_dir / f"{stem}.json"
        if not expected_file.exists():
            print(f"SKIP {stem}: no expected file")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(input_files)}] {stem}")
        print(f"{'=' * 60}")

        content = extract_text_from_file(input_file)
        content = clean_text(content)
        content = truncate_to_tokens(content, max_chars=30000)

        extraction = agent.run(
            context=content,
            schema=config.expected_schema,
            instructions=config.extraction_instructions,
            few_shot_examples=examples,
        )

        with open(expected_file) as f:
            expected = json.load(f)

        diffs = deep_compare(expected, extraction)

        fields_checked = 0
        fields_matched = 0

        def count_leaf_fields(d, depth=0):
            count = 0
            if isinstance(d, dict):
                for v in d.values():
                    if isinstance(v, (dict, list)):
                        count += count_leaf_fields(v, depth + 1)
                    else:
                        count += 1
            elif isinstance(d, list):
                for item in d:
                    if isinstance(item, dict):
                        count += count_leaf_fields(item, depth + 1)
                    else:
                        count += 1
            return count

        fields_checked = count_leaf_fields(expected)
        fields_matched = fields_checked - len(
            [d for d in diffs if d["type"] != "extra_field"]
        )

        total_fields_checked += fields_checked
        total_fields_matched += fields_matched
        total_diffs += len(diffs)

        accuracy = fields_matched / fields_checked * 100 if fields_checked > 0 else 0

        print(f"Fields: {fields_matched}/{fields_checked} matched ({accuracy:.1f}%)")
        print(f"Diffs: {len(diffs)}")

        for diff in diffs:
            if diff["type"] == "missing_field":
                print(f"  MISSING {diff['path']}: expected={diff['expected']!r}")
            elif diff["type"] == "extra_field":
                print(f"  EXTRA   {diff['path']}: actual={diff['actual']!r}")
            elif diff["type"] == "value_mismatch":
                exp_str = repr(diff["expected"])
                act_str = repr(diff["actual"])
                if len(exp_str) > 80:
                    exp_str = exp_str[:80] + "..."
                if len(act_str) > 80:
                    act_str = act_str[:80] + "..."
                print(f"  MISMATCH {diff['path']}:")
                print(f"    expected: {exp_str}")
                print(f"    actual:   {act_str}")
            elif diff["type"] == "missing_item":
                print(f"  MISSING_ITEM {diff['path']}: expected={diff['expected']!r}")
            elif diff["type"] == "extra_item":
                print(f"  EXTRA_ITEM {diff['path']}: actual={diff['actual']!r}")

        result_path = Path("data/evaluation_results")
        result_path.mkdir(parents=True, exist_ok=True)
        with open(result_path / f"{stem}_actual.json", "w") as f:
            json.dump(extraction, f, indent=2, default=str)

        time.sleep(1)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total documents: {len(input_files)}")
    print(f"Total diffs: {total_diffs}")
    print(
        f"Overall field accuracy: {total_fields_matched}/{total_fields_checked} ({total_fields_matched / total_fields_checked * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
