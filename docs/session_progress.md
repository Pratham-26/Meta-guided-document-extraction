# Session Progress

## Goal

Iteratively build the **best self-improving document extraction system using agents**. The system uses ColPali/ColBERT retrieval, DSPy prompt optimization (GEPA), and LangGraph orchestration to autonomously extract data from documents and continuously improve its own performance. We build in layers, validate with tests at every step, and refine the architecture as we learn what works. All tests must mock LLM calls.

## Instructions

- Follow the recommended build order from `technical_documentation.md`: config → schemas → storage → retrieval → agents → orchestration → optimization
- Use **uv** for all package management (`uv sync`, `uv run`)
- All tests must **mock LLM calls** — no real API calls during testing
- Follow architecture defined in `docs/architecture.md` (source of truth)
- Follow file structure in `docs/technical_documentation.md`
- Packages: langgraph, llama-index, dspy, pydantic, colpali-engine, ragatouille, python-dotenv, pdf2image
- System must be **LLM-agnostic** via DSPy + LiteLLM
- Bootstrapping requires **minimum 2 sample documents** per category

## Discoveries

1. **ragatouille** max version on PyPI is `0.0.9.post2` — do NOT pin `>=0.3`
2. **colpali-engine** also has version constraints — don't over-pin versions
3. **hatchling** needs explicit `[tool.hatch.build.targets.wheel] packages = ["src"]` since our package dir is `src/` not matching the project name
4. `Settings.categories_dir` and `Settings.traces_dir` are **computed properties** derived from `data_dir` — cannot monkeypatch them directly; only patch `data_dir`
5. DSPy agents use `dspy.Predict(Signature)` pattern — to mock LLM calls, mock `dspy.Predict` at the class level and return fake prediction objects with the expected output field attributes
6. **CRITICAL BUG — DSPy Signature field name conflict**: DSPy's `Signature` base class has `instructions` and `schema` as class attributes. Using these as field names in signature subclasses causes `TypeError: Field 'instructions' in 'ScoutExplore' must be declared with InputField or OutputField, but field 'instructions' has 'field.json_schema_extra=None'`. **Fix: rename** `schema` → `extraction_schema` and `instructions` → `extraction_instructions` in ALL DSPy Signature classes and their callers.
7. **Windows `os.rename` bug**: `Path.rename()` on Windows raises `FileExistsError` when target exists. **Fix**: use `os.replace()` in `_atomic_write()` in `src/storage/fs_store.py`
8. **PromptMutator naming conflict**: `self.mutate` instance attribute shadowed the `mutate()` method, making the method unreachable. **Fix**: renamed attribute to `self._mutate_prompt`
9. **`settings.configs_dir` is static** `Path("configs")` — not derived from `data_dir`. Tests that need category configs must also `patch.object(settings, "configs_dir", ...)` separately.
10. **`get_lm` import patching**: When patching `get_lm`, must patch at the usage site (e.g., `src.orchestration.nodes.get_lm`) since `nodes.py` imports it at module level with `from src.config.lm import get_lm`.
11. **Lazy-imported module mocking**: When source code uses `from X import Y` inside a function (lazy import), patching `module.Y` at the module level fails with `AttributeError`. **Fix**: use `patch.dict("sys.modules", {"X": mock_X})` to inject the mock before the lazy import runs. Also, combined imports like `import torch, pickle` require mocking both modules in `sys.modules`.
12. **Vision/text model duality for scout and extractor**: The scout and extractor agents must handle both PDF inputs (processed as images via ColPali) and text inputs (processed via ColBERT). Each dual-model role is configured with `text_model` and `vision_model` in `model_config.json`. `AgentRoleConfig.get_model(input_type)` resolves the correct model string at runtime. Judge and reflector use a single `model` since they only process text.

## Accomplished

### Completed

1. **Full project scaffold** — all directories from the technical docs created
2. **Layer 1: `src/config/`** — `settings.py`, `loader.py`, `lm.py`, `configs/model_config.json`
3. **Layer 2: `src/schemas/`** — all 6 Pydantic models
4. **Layer 3: `src/storage/`** — `paths.py`, `fs_store.py`, `trace_logger.py`
5. **Layer 4: `src/retrieval/`** — `router.py`, `colpali/`, `colbert/`
6. **Layer 5: `src/agents/`** — `scout/`, `extractor/`, `judge/`
7. **Layer 6: `src/orchestration/`** — `graph.py`, `state.py`, `nodes.py`, `hitl.py`
8. **Layer 7: `src/optimization/`** — `gepa.py`, `reflector.py`, `population.py`, `validator.py`
9. **Layer 8: `src/utils/`** + `scripts/` + `.env.example` + `pyproject.toml`
10. **DSPy Signature field rename** — `schema` → `extraction_schema`, `instructions` → `extraction_instructions` in all DSPy Signature classes and callers (scout, extractor, judge, reflector)
11. **Bug fixes** — `os.replace()` for Windows atomic writes; `self._mutate_prompt` to fix PromptMutator naming conflict
12. **132 unit tests passing, 7 failing** — 4 failures in `test_graph.py` (integration happy path + error halt), 2 in `test_scout.py` (explore/extraction), 1 other
13. **Integration test for full LangGraph pipeline** — happy path + error halt scenarios in `test_graph.py` (currently failing)
14. **Retrieval test fixes** — fixed mock patching for lazy-imported modules (`ragatouille`, `colpali_engine`, `pdf2image`) using `sys.modules` dict patching; fixed ColPali `pickle`/`torch` mock for `build_index`

### Test Coverage

| Test File | Tests | Covers |
|-----------|-------|--------|
| `test_schemas.py` | 10 | All 6 Pydantic models |
| `test_lm_config.py` | 5 | ModelConfig, CategoryConfig loading |
| `test_router.py` | 4 | InputType routing to COLPALI/COLBERT |
| `test_fs_store.py` | 8 | GoldStandard CRUD, QuestionStore, has_context |
| `test_scout.py` | 6 | ScoutAgent explore/infer, GoldBuilder, QuestionStore |
| `test_judge.py` | 3 | JudgeEvaluation schema validation |
| `test_judge_agent.py` | 6 | JudgeAgent.evaluate with mocked dspy.Predict (quality tiers, field diffs, confidence clamping) |
| `test_extractor.py` | 5 | ExtractorAgent.run with mocked dspy.Predict (JSON parsing, few-shot, schema/instructions passthrough) |
| `test_reflector.py` | 4 | Reflector.analyze + PromptMutator.mutate with mocked dspy.Predict |
| `test_population.py` | 13 | PromptCandidate model, CRUD, current prompt, next ID, Pareto selection |
| `test_gepa.py` | 4 | Full GEPA cycle (no gold standards error, initial candidate, multi-generation, skip empty samples) |
| `test_orchestration_nodes.py` | 12 | Pipeline nodes (check_context, resolve_config, load_questions, route_input, extract, judge, log_traces) |
| `test_validator.py` | 4 | Validator.validate_candidate (no gold standards, mocked agents, sample_size limit, zero accuracy) |
| `test_trace_logger.py` | 7 | Trace logging (single/batch write, phase dirs, read by date, empty reads) |
| `test_colpali_retriever.py` | 5 | ColPali build_index, retrieve (page indices, dedup), get_retrieved_pages (range skip) |
| `test_colbert_retriever.py` | 7 | ColBERT build_index, retrieve (dedup, top_k), get_retrieved_chunks (format, document key fallback) |
| `test_hitl.py` | 6 | Human-in-the-loop (present_for_review, apply_corrections — extraction, questions, both, empty) |
| `test_graph.py` | 10 | LangGraph pipeline (node existence, routing, error halt, compile, integration happy path + error) |
| **Total** | **139** (132 pass, 7 fail) | |

### Source Code Bug Fixes Applied

1. `src/storage/fs_store.py` — `Path.rename()` → `os.replace()` (Windows compatibility)
2. `src/optimization/reflector.py` — `self.mutate` → `self._mutate_prompt` (naming conflict fix)

### Still To Do

- End-to-end integration test with a real (small) document
- Iteration 2: evaluate extraction quality against real documents, identify weaknesses
- Iteration N: refine retrieval, prompts, and optimization based on measured performance

## Relevant Files and Directories

### Source code (all created and tested)

- `src/config/` — `settings.py`, `loader.py`, `lm.py`
- `src/schemas/` — `document.py`, `category.py`, `extraction.py`, `gold_standard.py`, `trace.py`, `evaluation.py`
- `src/storage/` — `paths.py`, `fs_store.py`, `trace_logger.py`
- `src/retrieval/` — `router.py`, `colpali/`, `colbert/`
- `src/agents/` — `scout/`, `extractor/`, `judge/`
- `src/orchestration/` — `graph.py`, `state.py`, `nodes.py`, `hitl.py`
- `src/optimization/` — `gepa.py`, `population.py`, `validator.py`
- `src/utils/` — `pdf.py`, `text.py`, `logging.py`

### Config

- `pyproject.toml`, `configs/model_config.json`, `.env.example`, `.gitignore`

### Tests (132 total)

- `tests/conftest.py` — comprehensive fixtures
- `tests/unit/test_schemas.py`, `test_lm_config.py`, `test_router.py`, `test_fs_store.py`
- `tests/unit/test_scout.py`, `test_judge.py`, `test_judge_agent.py`
- `tests/unit/test_extractor.py`, `test_reflector.py`
- `tests/unit/test_population.py`, `test_gepa.py`
- `tests/unit/test_orchestration_nodes.py`, `test_graph.py`, `test_hitl.py`
- `tests/unit/test_validator.py`, `test_trace_logger.py`
- `tests/unit/test_colpali_retriever.py`, `test_colbert_retriever.py`

### Scripts

- `scripts/bootstrap_category.py`, `run_scout.py`, `run_extraction.py`, `run_optimization.py`, `export_traces.py`

### Documentation (reference, not modified)

- `docs/architecture.md`
- `docs/technical_documentation.md`
- `docs/research/requirements.txt`
- `docs/research/package_research.md`

## Key Mock Patterns (from conftest.py)

```python
# Mock DSPy.Predict for agent tests
@pytest.fixture
def mock_dspy_predict():
    with patch("dspy.Predict") as mock_cls:
        yield mock_cls

# Usage: set return values with output field attributes
fake_result = MagicMock()
fake_result.extraction = '{"name": "Acme Corp"}'
mock_dspy_predict.return_value = MagicMock(return_value=fake_result)

# Mock full agent methods for orchestration/integration tests
@pytest.fixture
def mock_extractor_run():
    with patch("src.agents.extractor.agent.ExtractorAgent.run") as mock:
        mock.return_value = {"name": "Acme Corp", "amount": 5000.00}
        yield mock

# Patch get_lm at usage site (not definition site)
patch("src.orchestration.nodes.get_lm")  # correct
patch("src.config.lm.get_lm")            # wrong for modules that import it

# Mock lazy-imported heavy dependencies via sys.modules
with patch.dict("sys.modules", {"ragatouille": MagicMock(RAGPretrainedModel=mock_cls)}, clear=False):
    from src.retrieval.colbert.indexer import build_index
    build_index("test_category", documents)

# For combined lazy imports (import torch, pickle), mock both
with patch.dict("sys.modules", {"torch": MagicMock(), "pickle": mock_pickle}, clear=False):
    import importlib
    importlib.reload(indexer_mod)  # reload needed if module was already loaded
```
