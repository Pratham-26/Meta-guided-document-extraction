import argparse
import json
from pathlib import Path

from src.config.settings import settings
from src.config.loader import CategoryConfig
from src.storage import paths, ensure_category_dirs


def main():
    parser = argparse.ArgumentParser(description="Bootstrap a new document category")
    parser.add_argument(
        "--config", required=True, help="Path to category JSON config file"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config without writing"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    with open(config_path) as f:
        raw = json.load(f)

    config = CategoryConfig(**raw)

    if len(config.sample_documents) < 2:
        print("Error: At least 2 sample documents required for bootstrapping.")
        return

    for doc_path in config.sample_documents:
        if not Path(doc_path).exists():
            print(f"Warning: Sample document not found: {doc_path}")

    if args.dry_run:
        print(f"Dry run: Config for '{config.category_name}' is valid.")
        print(
            f"  Schema fields: {list(config.expected_schema.get('properties', {}).keys())}"
        )
        print(f"  Sample documents: {config.sample_documents}")
        return

    dest = settings.configs_dir / "categories" / f"{config.category_name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(raw, indent=2))
    print(f"Category config saved to {dest}")

    ensure_category_dirs(config.category_name)
    print(f"Directory structure created at {paths.category_dir(config.category_name)}")

    for doc_path in config.sample_documents:
        src = Path(doc_path)
        if src.exists():
            from src.storage.fs_store import save_source_document

            modality = "pdf" if src.suffix.lower() == ".pdf" else "text"
            saved = save_source_document(config.category_name, modality, src)
            print(f"  Copied: {src.name} -> {saved}")

    print(f"\nCategory '{config.category_name}' bootstrapped successfully.")
    print("Next step: Run the Scout Agent to build the knowledge base:")
    print(f"  python scripts/run_scout.py --category {config.category_name}")


if __name__ == "__main__":
    main()
