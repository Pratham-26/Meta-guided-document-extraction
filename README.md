# Meta-Learning for Document Extraction

An autonomous, self-optimizing document extraction system that uses multi-agent LLM pipelines to extract structured data from documents, evaluate quality against gold standards, and continuously improve its extraction instructions through evolutionary optimization.

## How It Works

The system uses a **LangGraph-orchestrated pipeline** with four specialized DSPy agents:

1. **Scout Agent** — Explores documents (text or PDF) to build gold standards and infer retrieval questions. Supports vision models for PDF page analysis, chunked processing for large documents, and RLM (Recursive Language Model) for oversized contexts.

2. **Extractor Agent** — Performs structured extraction using retrieved context, schema definitions, and few-shot examples from approved gold standards. Supports both text and vision inputs.

3. **Judge Agent** — Compares extractions against approved gold standards, producing quality tiers (low/medium/high), field-level diffs, and confidence scores.

4. **Reflector Agent** — Analyzes extraction failures, diagnoses instruction-level root causes, and generates mutated prompt candidates for optimization.

The pipeline also uses **ColBERT** (for text) and **ColPali** (for PDF) retrieval to find the most relevant chunks/pages before extraction.

### Self-Optimization Loop (GEPA)

The system runs a **Gene Expression Programming for Adapters (GEPA)** cycle:

- Identifies low/medium quality extractions from trace logs
- Uses the Reflector to diagnose failures and propose instruction fixes
- Generates a population of mutated prompt candidates
- Validates each candidate against gold standards
- Selects the best via Pareto front selection
- Promotes the winner as the active extraction instructions

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install

```bash
git clone <repo-url>
cd Meta-learning-for-document-extraction
uv sync
```

### Configure

```bash
cp .env.example .env
# Edit .env with your API keys (LiteLLM-compatible providers)
```

Set your LLM provider keys in `.env`. The system uses LiteLLM, so any provider works (OpenAI, Anthropic, Gemini, Azure, local Ollama, etc.). Model assignments are configured in `configs/model_config.json`.

### Workflow

#### 1. Bootstrap a Category

```bash
python scripts/bootstrap_category.py --config configs/categories/service_agreements.json
```

This validates the category config (schema + sample documents) and creates the directory structure. The JSON config defines the extraction schema, instructions, and sample document paths.

#### 2. Run the Scout Agent

```bash
python scripts/run_scout.py --category service_agreements
```

The Scout explores all sample documents, builds gold standards, infers retrieval questions, and rebuilds the search index. Gold standards start as `PENDING_REVIEW` by default.

Add `--auto-approve` to skip human review, or use the review server:

```bash
python scripts/review_server.py --port 8111
```

Open `http://localhost:8111` to review, edit, approve, or reject gold standards.

#### 3. Run Extraction

```bash
python scripts/run_extraction.py --document path/to/doc.txt --category service_agreements
```

For gold standard candidates:

```bash
python scripts/run_extraction.py --document path/to/doc.txt --category service_agreements --gold
```

#### 4. Run Optimization

```bash
python scripts/run_optimization.py --category service_agreements --modality text
```

Override defaults with `--generations` and `--population`.

#### 5. Evaluate & Export

Batch evaluate against expected outputs:

```bash
python scripts/batch_evaluate.py
```

Export trace logs for fine-tuning:

```bash
python scripts/export_traces.py --category service_agreements --modality text --output data/traces_export
```

## Project Structure

```
configs/
  categories/           # Category configs (schema + instructions + sample docs)
  model_config.json     # LLM model assignments per agent role
  process_config.json   # Gold sampling rate and auto-gold settings
examples/               # Sample documents and expected outputs
scripts/
  bootstrap_category.py # Bootstrap a new document category
  run_scout.py          # Run Scout agent to build knowledge base
  run_extraction.py     # Run extraction pipeline on a document
  run_optimization.py   # Run GEPA optimization cycle
  batch_evaluate.py     # Batch evaluate against expected outputs
  review_server.py      # HITL gold standard review UI
  export_traces.py      # Export traces for SLM training
src/
  agents/
    scout/              # Scout agent: exploration, gold building, question inference
    extractor/          # Extractor agent: structured extraction with few-shot
    judge/              # Judge agent: quality evaluation with field diffs
  config/               # Settings, model config, category config loader
  orchestration/        # LangGraph pipeline: state, nodes, graph
  optimization/         # GEPA: reflector, population, validator
  retrieval/
    colbert/            # ColBERT indexer and retriever (text)
    colpali/            # ColPali indexer and retriever (PDF)
  schemas/              # Pydantic models: documents, gold standards, evaluations, traces
  storage/              # File-based storage, trace logging, paths
  utils/                # PDF/text extraction, logging
tests/
  unit/                 # Unit tests for all components
  integration/          # Integration tests for the pipeline
```

## Configuration

### Category Config (`configs/categories/<name>.json`)

Defines the extraction schema (JSON Schema), extraction instructions, and sample document paths. See `configs/categories/service_agreements.json` for a full example.

### Model Config (`configs/model_config.json`)

Assigns LLM models per agent role with temperature and token limits. Each role can have separate `text_model` and `vision_model`.

### Process Config (`configs/process_config.json`)

- `gold_sampling_rate` — How often non-gold documents are sampled for gold generation (default: every 100)
- `auto_gold_initial_count` — First N documents are automatically treated as gold candidates (default: 10)

## Running Tests

```bash
uv run pytest
```

## Tech Stack

- **DSPy** — LLM programming framework for agents and signatures
- **LangGraph** — Pipeline orchestration and state management
- **Pydantic** — Schema validation and settings
- **ColPali Engine** — Visual retrieval for PDF documents
- **RAGatouille** — ColBERT indexing and retrieval for text
- **LiteLLM** — Unified interface for any LLM provider

## License

Private repository. All rights reserved.
