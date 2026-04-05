# AGENTS.md

## Commands

```bash
# Run all tests
pytest

# Single test file / class / function
pytest tests/unit/test_scout.py
pytest tests/unit/test_scout.py::TestGoldBuilder::test_build_and_save
pytest tests/unit/test_scout.py -k "test_small_content"

# Only unit or integration
pytest tests/unit/
pytest tests/integration/

# Run scripts (from repo root)
python scripts/run_extraction.py --document <path> --category <name> [--gold]
python scripts/run_scout.py --category <name> [--doc <path> ...]
python scripts/run_optimization.py --category <name> --modality pdf|text
python scripts/bootstrap_category.py --config <path>
python scripts/batch_evaluate.py  # hardcoded to service_agreements
python scripts/export_traces.py --category <name> --modality pdf|text --output <dir>
```

No linter or formatter is configured. No CI workflows exist.

## Architecture

LangGraph pipeline in `src/orchestration/` wires these nodes in order:
`check_context → resolve_config → detect_gold → [run_scout_for_gold] → load_questions → route_input → retrieve → extract → [judge] → log_traces → cleanup_index`

- **Gold documents** (user-flagged or auto-sampled) branch through `run_scout_for_gold` and `judge`; regular docs skip both.
- `PipelineState` is a `TypedDict`, not a Pydantic model.

### Three DSPy agents

| Agent | Role | Location |
|-------|------|----------|
| Scout | Explore docs, build gold standards, infer retrieval questions | `src/agents/scout/` |
| Extractor | Structured extraction from retrieved context | `src/agents/extractor/` |
| Judge | Compare extraction vs gold standard, assign quality tier | `src/agents/judge/` |

### Retrieval routing

`src/retrieval/router.py` routes PDFs to **ColPali** and text to **ColBERT**.

### GEPA optimization

`src/optimization/gepa.py` runs evolutionary prompt optimization using a Reflector + PromptMutator, validated against gold standards.

## Key patterns

- **DSPy LM caching**: `src/config/lm.py` caches `dspy.LM` instances per `role:input_type`. Agents call `dspy.configure(lm=...)` which sets a global — be careful with concurrent usage.
- **RLM threshold**: Scout uses `dspy.RLM` for documents >32K estimated tokens (4 chars/token heuristic), else `dspy.Predict`.
- **Vision fallback**: When PyMuPDF text extraction returns a placeholder string starting with `[PDF with`, the system loads page images and uses the vision model instead.
- **ColBERT compatibility patches**: `src/retrieval/colbert_compat.py` monkey-patches langchain, torch extensions, and tied-weights issues. Must call `ensure_colbert_compat()` before ColBERT imports.
- **Model naming**: Uses LiteLLM format (`provider/model`, e.g. `zai/glm-4.7-flash`, `openai/gpt-4o`) in `configs/model_config.json`.

## Testing conventions

- **pytest config** (`pyproject.toml`): `pythonpath = ["."]` — imports are `src.xxx`.
- **DSPy is heavily mocked**: `conftest.py` provides `mock_dspy_predict`, `mock_dspy_rlm`, `mock_dspy_lm`, `mock_get_lm`, plus per-agent mock fixtures (`mock_scout_explore`, `mock_extractor_run`, `mock_judge_evaluate`, etc.).
- **Filesystem isolation**: `tmp_category_dir` fixture monkeypatches `settings.data_dir` to a temp path.
- **Integration tests** (`tests/integration/test_pipeline.py`) mock LLM/retrieval calls but exercise the real LangGraph graph with real config loading and file I/O.

## Config and data

| Path | Purpose |
|------|---------|
| `configs/model_config.json` | Model assignments per agent role |
| `configs/process_config.json` | Gold sampling rate, auto-gold initial count |
| `configs/categories/<name>.json` | Per-category: schema, extraction instructions, sample docs |
| `.env` | API keys (LiteLLM auto-reads `OPENAI_API_KEY`, etc.) |
| `data/` | Runtime data (gitignored): categories, gold standards, traces, indexes |

### Workflow to add a new document category

1. Create a category JSON config (see `configs/categories/service_agreements.json` for reference; must have `category_name`, `expected_schema`, `extraction_instructions`, and ≥2 `sample_documents`).
2. `python scripts/bootstrap_category.py --config <path>` — copies config, creates directory structure.
3. `python scripts/run_scout.py --category <name>` — explores sample docs, builds gold standards, infers questions, rebuilds index.
4. Run extractions with `python scripts/run_extraction.py --document <path> --category <name>`.
