# Technical Documentation: Meta-Guided Document Extraction System

> **Version:** 0.1.0 — Pre-Implementation  
> **Last Updated:** 2026-04-03  
> **Status:** Planning / Bootstrapping  
> **Reference:** [Architecture Document](./architecture.md)

---

## Table of Contents

1. [Project File Structure](#1-project-file-structure)
2. [Module Reference](#2-module-reference)
3. [Configuration Reference](#3-configuration-reference)
4. [Data Models & Schemas](#4-data-models--schemas)
5. [Data Flow & Interfaces](#5-data-flow--interfaces)
6. [Storage Layout](#6-storage-layout)
7. [Development Setup](#7-development-setup)
8. [Testing Strategy](#8-testing-strategy)
9. [Deployment & Operations](#9-deployment--operations)

---

## 1. Project File Structure

```
Meta-learning-for-document-extraction/
│
├── .env                              # Local environment variables (secrets, API keys)
├── .env.example                      # Template for required environment variables
├── .gitignore
├── pyproject.toml                    # Project metadata, dependencies, build config
├── README.md                         # Project overview & quickstart
│
├── docs/                             # Documentation
│   ├── architecture.md               # System architecture (source of truth)
│   ├── technical_documentation.md    # This file — implementation reference
│   └── research/
│       ├── requirements.txt          # Core package list
│       └── package_research.md       # Per-package rationale
│
├── configs/                          # JSON configuration files
│   ├── model_config.json             # DSPy LM configuration (LiteLLM model strings)
│   └── categories/                   # Per-document-category config
│       └── commercial_lease.json     # Example: schema, instructions, retrieval params
│
├── src/                              # Source code root
│   ├── __init__.py
│   │
│   ├── config/                       # Configuration loading & validation
│   │   ├── __init__.py
│   │   ├── settings.py               # Global settings (env vars, paths, defaults)
│   │   ├── loader.py                 # JSON config loader with Pydantic validation
│   │   └── lm.py                     # DSPy LM initialization (wraps dspy.LM with LiteLLM)
│   │
│   ├── schemas/                      # Pydantic data models
│   │   ├── __init__.py
│   │   ├── document.py               # Document input models (PDF, text payloads)
│   │   ├── category.py               # Category definition schema
│   │   ├── extraction.py             # Extraction output schema (dynamic per category)
│   │   ├── gold_standard.py          # Gold Standard record schema
│   │   ├── trace.py                  # Trace log entry schema
│   │   └── evaluation.py            # Judge evaluation result schema
│   │
│   ├── agents/                       # Agent implementations
│   │   ├── __init__.py
│   │   ├── scout/                    # Scout Agent (Knowledge Base Builder)
│   │   │   ├── __init__.py
│   │   │   ├── agent.py              # DSPy RLM agent — REPL loop, question inference
│   │   │   ├── question_store.py     # Manage inferred questions per category
│   │   │   └── gold_builder.py       # Gold Standard construction from Scout output
│   │   │
│   │   ├── extractor/                # Extraction Agent
│   │   │   ├── __init__.py
│   │   │   ├── agent.py              # DSPy module — extraction with compiled prompts
│   │   │   └── few_shot.py           # Few-shot example selection & synthesis
│   │   │
│   │   └── judge/                    # Judge Agent
│   │       ├── __init__.py
│   │       └── agent.py              # Comparison function — extraction vs Gold Standard
│   │
│   ├── retrieval/                    # Retrieval Layer
│   │   ├── __init__.py
│   │   ├── router.py                 # Input-type router (PDF → ColPali, text → ColBERT)
│   │   ├── colpali/                  # Vision retrieval
│   │   │   ├── __init__.py
│   │   │   ├── indexer.py            # Page-level visual embedding & index management
│   │   │   └── retriever.py          # Question-driven page retrieval
│   │   │
│   │   └── colbert/                  # Text retrieval
│   │       ├── __init__.py
│   │       ├── indexer.py            # Token-level text embedding & index management
│   │       └── retriever.py          # Question-driven chunk retrieval (MaxSim)
│   │
│   ├── orchestration/                # LangGraph Orchestration Layer
│   │   ├── __init__.py
│   │   ├── graph.py                  # Main LangGraph state graph definition
│   │   ├── state.py                  # Graph state schema (TypedDict / Pydantic)
│   │   ├── nodes.py                  # Graph node functions (route, extract, evaluate)
│   │   └── hitl.py                   # Human-in-the-Loop breakpoint handlers
│   │
│   ├── optimization/                 # Continuous Learning / GEPA Layer
│   │   ├── __init__.py
│   │   ├── gepa.py                   # GEPA optimizer — evolutionary prompt refinement
│   │   ├── reflector.py              # Reflection LLM — failure diagnosis
│   │   ├── population.py             # Prompt candidate population management
│   │   └── validator.py              # Post-optimization validation runner
│   │
│   ├── storage/                      # Persistent Storage Layer
│   │   ├── __init__.py
│   │   ├── fs_store.py               # File-system store for Gold Standards & documents
│   │   ├── trace_logger.py           # Structured trace logging (all LLM interactions)
│   │   └── paths.py                  # Canonical path resolution for all stored artifacts
│   │
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       ├── pdf.py                    # PDF loading, page rendering to images
│       ├── text.py                   # Text preprocessing, chunking helpers
│       └── logging.py               # Application-level logging configuration
│
├── data/                             # Runtime data root (gitignored except structure)
│   ├── categories/                   # Per-category runtime data
│   │   └── <category_name>/
│   │       ├── gold_standards/       # Approved Gold Standard JSON + source docs
│   │       │   ├── gs_001.json
│   │       │   └── sources/
│   │       │       └── lease_001.pdf
│   │       ├── questions/            # Scout-inferred questions
│   │       │   └── questions.json
│   │       ├── indexes/              # ColPali / ColBERT index artifacts
│   │       │   ├── colpali/
│   │       │   └── colbert/
│   │       └── prompts/              # GEPA prompt population & current winner
│   │           ├── current.json
│   │           └── population/
│   │               ├── candidate_001.json
│   │               └── candidate_002.json
│   │
│   └── traces/                       # Structured trace logs (SLM training corpus)
│       └── <category_name>/
│           ├── extraction_traces/
│           │   └── trace_2026-04-03_001.jsonl
│           ├── judge_traces/
│           │   └── trace_2026-04-03_001.jsonl
│           └── optimization_traces/
│               └── trace_2026-04-03_001.jsonl
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures (mock LLMs, sample docs)
│   ├── unit/
│   │   ├── test_schemas.py
│   │   ├── test_lm_config.py
│   │   ├── test_router.py
│   │   ├── test_fs_store.py
│   │   └── test_judge.py
│   ├── integration/
│   │   ├── test_scout_pipeline.py
│   │   ├── test_extraction_pipeline.py
│   │   └── test_gepa_cycle.py
│   └── fixtures/
│       ├── sample_lease.pdf
│       ├── sample_text_doc.txt
│       └── sample_gold_standard.json
│
└── scripts/                          # Utility & operations scripts
    ├── bootstrap_category.py         # CLI: initialize a new document category
    ├── run_scout.py                  # CLI: trigger Scout Agent for a category
    ├── run_extraction.py             # CLI: run extraction on a document
    ├── run_optimization.py           # CLI: trigger GEPA optimization cycle
    └── export_traces.py             # CLI: export trace logs for SLM training
```

---

## 2. Module Reference

### 2.1 `src/config/` — Configuration Management

| File | Responsibility |
|:---|:---|
| `settings.py` | Loads environment variables via `os.environ` / `python-dotenv`. Defines global defaults: data directory paths, logging levels. |
| `loader.py` | Reads JSON config files from `configs/`. Validates them against Pydantic models. Returns typed config objects for categories and model configuration. |
| `lm.py` | Initializes DSPy language models using `dspy.LM()`. Reads `configs/model_config.json` and sets up per-agent LM instances. All LLM access flows through this module. |

**Key design rule:** No module outside `src/config/` should read environment variables or raw JSON configs directly. All configuration flows through `settings.py` and `loader.py`.

---

### 2.2 LLM Provider Management — DSPy + LiteLLM

The system is **LLM-agnostic** — but we achieve this through **DSPy's built-in LiteLLM integration**, not a custom abstraction layer.

**Why no custom provider layer?** DSPy uses [LiteLLM](https://docs.litellm.ai/) internally as its LLM router. LiteLLM already provides:
- **100+ provider support** — OpenAI, Anthropic, Google, Cohere, Azure, Ollama, vLLM, HuggingFace, and more — all via a unified `model` string.
- **Automatic API key resolution** — reads standard environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) with zero config.
- **Consistent interface** — every provider responds with the same structure regardless of backend.
- **Cost tracking, retries, fallbacks** — built in.

This means the entire provider layer is a thin config file + a few lines of DSPy initialization:

```python
# src/config/lm.py — All LLM management

import dspy
import json
from pathlib import Path

def load_model_config(config_path: Path = Path("configs/model_config.json")) -> dict:
    """Load per-agent model configuration."""
    with open(config_path) as f:
        return json.load(f)

def get_lm(agent_role: str) -> dspy.LM:
    """
    Return a configured DSPy LM for the given agent role.
    Uses LiteLLM model strings — any supported provider works.
    """
    config = load_model_config()
    role_config = config["agent_roles"][agent_role]

    return dspy.LM(
        model=role_config["model"],           # LiteLLM model string
        temperature=role_config.get("temperature", 0.0),
        max_tokens=role_config.get("max_tokens", 4096),
    )

# Usage in any agent:
# lm = get_lm("scout")
# dspy.configure(lm=lm)
```

**LiteLLM model string format:** The `model` field uses LiteLLM's unified naming convention:

| Provider | Model String Example |
|:---|:---|
| OpenAI | `openai/gpt-4o`, `openai/gpt-4o-mini` |
| Anthropic | `anthropic/claude-sonnet-4-20250514` |
| Google | `gemini/gemini-2.0-flash` |
| Ollama (local) | `ollama/llama3.1:70b` |
| Azure OpenAI | `azure/my-deployment-name` |
| HuggingFace | `huggingface/meta-llama/Llama-3.1-8B` |

Swapping a provider is a one-line config change — no code modifications required.

---

### 2.3 `src/schemas/` — Data Models (Pydantic)

All data flowing through the system is strictly typed. Key schemas:

| Schema | Purpose |
|:---|:---|
| `DocumentInput` | Wraps incoming documents — file path, input type (`pdf` / `text`), raw content, metadata. |
| `CategoryConfig` | Defines a document category — name, expected output schema, extraction instructions, retrieval parameters. |
| `ExtractionResult` | The Extractor Agent's output — dynamically validated against the category's expected schema. |
| `GoldStandard` | An approved extraction — the source document ref, the approved JSON, who approved it, timestamp. |
| `TraceEntry` | A single LLM interaction — prompt, response, agent role, phase, timestamp, token counts. |
| `JudgeEvaluation` | Judge output — quality tier (`low` / `medium` / `high`), textual feedback, field-level diff. |

---

### 2.4 `src/agents/` — Agent Implementations

#### 2.4.1 Scout Agent (`agents/scout/`)

| File | Role |
|:---|:---|
| `agent.py` | DSPy RLM implementation. Runs iterative exploration over sample documents using DSPy's sandboxed REPL loop. Infers questions that define what needs to be extracted. For PDF inputs, the first pass processes pages as images to understand visual topography. |
| `question_store.py` | CRUD for the per-category question set. Questions are stored as JSON in `data/categories/<name>/questions/`. These questions become retrieval queries at extraction time. |
| `gold_builder.py` | Takes the Scout's exploratory output and structures it into a formal Gold Standard record. Persists to `data/categories/<name>/gold_standards/`. |

**Execution model:** Offline / periodic. Not on the live extraction path. Triggered via `scripts/run_scout.py` or scheduled.

#### 2.4.2 Extractor Agent (`agents/extractor/`)

| File | Role |
|:---|:---|
| `agent.py` | DSPy module wrapping the extraction prompt. Receives only the **retrieved pages/chunks** (never the full document). Uses compiled instructions + few-shot examples from GEPA optimization. Output is validated against the category's Pydantic schema. |
| `few_shot.py` | Selects the most relevant few-shot examples from the Gold Standard store for the current document category. GEPA can synthesize and replace these. |

#### 2.4.3 Judge Agent (`agents/judge/`)

| File | Role |
|:---|:---|
| `agent.py` | Comparison function. Takes an `ExtractionResult` and the corresponding `GoldStandard`, outputs a `JudgeEvaluation`. Only runs when a Gold Standard exists for the category. Assigns `low` / `medium` / `high` quality tier and generates textual feedback identifying divergences. |

---

### 2.5 `src/retrieval/` — Retrieval Layer

#### Router (`router.py`)
Straightforward input-type check:
- **PDF → ColPali** (vision route)
- **Text / text-heavy format → ColBERT** (text route)

No ML classifier — just file type detection.

#### ColPali (`retrieval/colpali/`)

| File | Role |
|:---|:---|
| `indexer.py` | Takes a PDF, renders each page to an image, generates visual patch embeddings via `colpali-engine`. Stores the index in `data/categories/<name>/indexes/colpali/`. |
| `retriever.py` | Takes the Scout's inferred questions as queries. Returns **only the relevant pages** ranked by visual similarity. |

#### ColBERT (`retrieval/colbert/`)

| File | Role |
|:---|:---|
| `indexer.py` | Takes text input, generates token-level embeddings via RAGatouille/ColBERT. Stores the index in `data/categories/<name>/indexes/colbert/`. |
| `retriever.py` | Takes the Scout's inferred questions as queries. Uses MaxSim operation to return **only the relevant chunks**. |

**Key principle:** The Extraction Agent never sees the full document. It only operates on pages (ColPali) or chunks (ColBERT) retrieved using the Scout's questions.

---

### 2.6 `src/orchestration/` — LangGraph State Machine

| File | Role |
|:---|:---|
| `state.py` | Defines the graph's `TypedDict` state: document input, category config, retrieved context, extraction draft, judge evaluation, optimization status. |
| `graph.py` | Builds the LangGraph `StateGraph`. Defines nodes and edges. Handles conditional routing (e.g., has Gold Standard → run Judge; no Gold Standard → skip). |
| `nodes.py` | Individual node functions — each wraps an agent or service call. Keeps graph definition clean. |
| `hitl.py` | Human-in-the-Loop interrupt handlers. Pauses graph execution, presents draft to user, resumes on approval/correction. |

**Graph Topology (simplified):**

```
START
  │
  ▼
[Input Router] ── PDF ──► [ColPali Retrieval]
       │                         │
       └── Text ──► [ColBERT Retrieval]
                         │
                         ▼
                  [Extraction Agent]
                         │
                  ┌──────┴──────┐
                  │ Gold        │ No Gold
                  │ Standard?   │ Standard
                  ▼             ▼
           [Judge Agent]    [HitL: Approval]
                  │              │
                  ▼              ▼
         [Log Trace]     [Promote to Gold]
                  │
                  ▼
               END
```

---

### 2.7 `src/optimization/` — GEPA Continuous Learning

| File | Role |
|:---|:---|
| `gepa.py` | Core optimizer. Consumes Judge evaluations. Orchestrates: reflection → mutation → selection → validation cycle. Runs offline (nightly batch or on-demand). |
| `reflector.py` | Reflection LLM call. Analyzes extraction traces alongside Gold Standards to diagnose *why* failures occurred. Produces structured failure reports. |
| `population.py` | Manages the prompt candidate population. Handles Pareto-based selection — maintains diversity so the system doesn't collapse into one suboptimal prompt. |
| `validator.py` | Post-optimization validation. Re-runs extraction on a representative mix of low/medium/high examples. Judge re-evaluates to confirm improvement. Only promotes the winning candidate on net improvement. |

**Optimization Cycle:**

```
Judge evaluations (low/medium) + Gold Standards
           │
           ▼
    ┌─────────────┐
    │  Reflector   │  ← diagnose failure causes
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   Mutator    │  ← targeted prompt revisions
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Population  │  ← Pareto selection across candidates
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Validator   │  ← re-run + re-evaluate
    └──────┬──────┘
           │
           ▼
    Deploy winning prompt → current.json
```

---

### 2.8 `src/storage/` — Persistent File System Store

| File | Role |
|:---|:---|
| `fs_store.py` | CRUD operations for Gold Standards, source documents, and indexes. All writes are atomic (write-to-temp → rename). Enforces the canonical directory layout under `data/`. |
| `trace_logger.py` | Appends structured trace entries (JSONL format) for every LLM interaction. Organized by category and phase. These logs form the future SLM training corpus. |
| `paths.py` | Single source of truth for path resolution. Every module that needs a file path calls `paths.py` — no hardcoded paths anywhere else. |

---

## 3. Configuration Reference

### 3.1 Environment Variables (`.env`)

LiteLLM (via DSPy) reads API keys from standard environment variables automatically. No custom env-var mapping needed.

| Variable | Required | Description |
|:---|:---|:---|
| `OPENAI_API_KEY` | If using OpenAI | Picked up automatically by LiteLLM |
| `ANTHROPIC_API_KEY` | If using Anthropic | Picked up automatically by LiteLLM |
| `GEMINI_API_KEY` | If using Google | Picked up automatically by LiteLLM |
| `AZURE_API_KEY` | If using Azure | Picked up automatically by LiteLLM |
| `OLLAMA_API_BASE` | If using Ollama | Local model server URL (default: `http://localhost:11434`) |
| `DATA_DIR` | No | Override data directory (default: `./data`) |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) |

> **Note:** LiteLLM supports 100+ providers. See the [LiteLLM provider docs](https://docs.litellm.ai/docs/providers) for the full list of supported environment variables per provider.

### 3.2 `configs/model_config.json`

Maps each agent role to a LiteLLM model string. Swapping providers is a one-line change — just update the model string.

```json
{
  "agent_roles": {
    "scout": {
      "model": "openai/gpt-4o",
      "temperature": 0.2,
      "max_tokens": 8192
    },
    "extractor": {
      "model": "openai/gpt-4o-mini",
      "temperature": 0.0,
      "max_tokens": 4096
    },
    "judge": {
      "model": "anthropic/claude-sonnet-4-20250514",
      "temperature": 0.0,
      "max_tokens": 4096
    },
    "reflector": {
      "model": "openai/gpt-4o",
      "temperature": 0.3,
      "max_tokens": 8192
    }
  }
}
```

**To switch to a local model**, just change the model string:
```json
"extractor": {
  "model": "ollama/llama3.1:70b",
  "temperature": 0.0
}
```
No code changes. No new provider implementations. LiteLLM handles it.

### 3.3 `configs/categories/<name>.json`

Per-document-category configuration. One file per category.

```json
{
  "category_name": "commercial_lease_agreement",
  "expected_schema": {
    "type": "object",
    "properties": {
      "landlord_name": { "type": "string", "description": "Full legal name of the landlord" },
      "tenant_name": { "type": "string", "description": "Full legal name of the tenant" },
      "monthly_rent": { "type": "number", "description": "Monthly rent in USD" },
      "lease_start_date": { "type": "string", "format": "date" },
      "lease_end_date": { "type": "string", "format": "date" },
      "security_deposit": { "type": "number" }
    },
    "required": ["landlord_name", "tenant_name", "monthly_rent"]
  },
  "extraction_instructions": "Extract the names of the signing parties and the financial terms including rent, deposit, and lease duration.",
  "retrieval": {
    "default_route": "auto",
    "colpali_top_k": 3,
    "colbert_top_k": 5
  },
  "optimization": {
    "gepa_population_size": 8,
    "gepa_generations": 5,
    "validation_sample_size": 10
  }
}
```

---

## 4. Data Models & Schemas

### 4.1 Core Pydantic Models

```python
# --- src/schemas/document.py ---
from pydantic import BaseModel
from enum import Enum
from pathlib import Path

class InputType(str, Enum):
    PDF = "pdf"
    TEXT = "text"

class DocumentInput(BaseModel):
    """Incoming document to be processed."""
    source_uri: Path                  # Path to the source file
    input_type: InputType             # Determines retrieval route
    category: str                     # Target category name
    raw_text: str | None = None       # Pre-extracted text (if available)
    metadata: dict = {}


# --- src/schemas/gold_standard.py ---
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path

class GoldStandard(BaseModel):
    """An approved, ground-truth extraction for a specific document."""
    id: str                           # Unique identifier (e.g., "gs_001")
    category: str
    source_document_uri: Path         # Path to the original document
    extraction: dict                  # The approved JSON extraction
    approved_by: str                  # "human" | "scout"
    created_at: datetime
    supersedes: str | None = None     # ID of the Gold Standard this replaces


# --- src/schemas/evaluation.py ---
from pydantic import BaseModel
from enum import Enum

class QualityTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class JudgeEvaluation(BaseModel):
    """Output of the Judge Agent's comparison."""
    quality_tier: QualityTier
    feedback: str                     # Textual explanation of divergences
    field_diffs: list[dict]           # Per-field comparison details
    gold_standard_id: str             # Which Gold Standard was compared against
    confidence: float                 # 0.0 – 1.0


# --- src/schemas/trace.py ---
from pydantic import BaseModel
from datetime import datetime

class TraceEntry(BaseModel):
    """A single logged LLM interaction."""
    timestamp: datetime
    agent_role: str                   # "scout" | "extractor" | "judge" | "reflector"
    phase: str                        # "bootstrap" | "extraction" | "evaluation" | "optimization"
    category: str
    prompt: str                       # Full prompt sent to LLM
    response: str                     # Full LLM response
    model: str                        # Model identifier
    provider: str                     # Provider name
    token_usage: dict                 # {"prompt_tokens": ..., "completion_tokens": ...}
    quality_tier: str | None = None   # If applicable (post-Judge)
    document_id: str | None = None    # Source document reference
```

---

## 5. Data Flow & Interfaces

### 5.1 Phase 1 — Bootstrapping Flow

```
User provides:  CategoryConfig JSON + sample document
                         │
                         ▼
              ┌─────────────────────┐
              │  bootstrap_category │  (scripts/)
              │  • Validate config  │
              │  • Save to configs/ │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  LangGraph: Boot    │
              │  • Route input      │
              │  • Retrieve context │
              │  • Extract (zero-   │
              │    shot)            │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  HitL Breakpoint    │
              │  • Present draft    │
              │  • Human corrects   │
              │  • Human approves   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Promote to Gold    │
              │  • Save Gold JSON   │
              │  • Archive source   │
              │  • Log trace        │
              └─────────────────────┘
```

### 5.2 Phase 1.5 — Scout Flow

```
Trigger:  Scheduled / manual per category
                         │
                         ▼
              ┌─────────────────────┐
              │  Scout Agent (RLM)  │
              │  • Load category    │
              │  • Explore samples  │
              │  • Image-first pass │
              │    (PDF)            │
              │  • REPL reasoning   │
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
     ┌──────────────┐    ┌───────────────────┐
     │ Questions     │    │ Gold Standard     │
     │ • Inferred Qs │    │ • Rigorous        │
     │ • Saved to    │    │   extraction      │
     │   questions/  │    │ • Supersedes      │
     └──────────────┘    │   Phase 1 gold    │
                         │ • Saved to gold_  │
                         │   standards/      │
                         └───────────────────┘
```

### 5.3 Phase 2 — Production Extraction Flow

```
New document arrives
         │
         ▼
  ┌──────────────┐
  │ Input Router  │
  └──────┬───────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 [ColPali] [ColBERT]
    │         │
    └────┬────┘
         │  Retrieved pages/chunks
         ▼
  ┌──────────────┐
  │  Extractor   │ ← compiled DSPy prompt + few-shot
  │  Agent       │
  └──────┬───────┘
         │  ExtractionResult
         ▼
  ┌──────────────┐
  │ Gold Standard│── No ──► Return result (no eval)
  │ exists?      │
  └──────┬───────┘
         │ Yes
         ▼
  ┌──────────────┐
  │ Judge Agent  │ ← compare against Gold Standard
  └──────┬───────┘
         │  JudgeEvaluation
         ▼
  ┌──────────────┐
  │ Log trace    │
  │ Return result│
  └──────────────┘
```

### 5.4 Phase 3 — GEPA Optimization Flow

```
Trigger: Nightly batch / on-demand
         │
         ▼
  Collect Judge evaluations (low + medium tier)
         │
         ▼
  ┌───────────────┐
  │ Reflector LLM │ ← analyze traces vs Gold Standards
  └──────┬────────┘
         │ Failure diagnoses
         ▼
  ┌───────────────┐
  │ Mutator       │ ← propose targeted prompt revisions
  └──────┬────────┘
         │ New candidates
         ▼
  ┌───────────────┐
  │ Population    │ ← Pareto selection
  └──────┬────────┘
         │ Top candidates
         ▼
  ┌───────────────┐
  │ Validator     │ ← re-extract + re-judge on mixed sample
  └──────┬────────┘
         │ Best candidate
         ▼
  Deploy → prompts/current.json
```

---

## 6. Storage Layout

### 6.1 Gold Standard File Format

```
data/categories/commercial_lease/gold_standards/
├── gs_001.json          ← Gold Standard metadata + extraction
├── gs_002.json
└── sources/
    ├── lease_001.pdf    ← Original source document (immutable)
    └── lease_002.pdf
```

**`gs_001.json` example:**
```json
{
  "id": "gs_001",
  "category": "commercial_lease_agreement",
  "source_document_uri": "sources/lease_001.pdf",
  "extraction": {
    "landlord_name": "Acme Properties LLC",
    "tenant_name": "Contoso Corp",
    "monthly_rent": 8500.00,
    "lease_start_date": "2026-01-01",
    "lease_end_date": "2028-12-31",
    "security_deposit": 17000.00
  },
  "approved_by": "scout",
  "created_at": "2026-04-03T10:30:00Z",
  "supersedes": null
}
```

### 6.2 Trace Log Format (JSONL)

Each line is a standalone JSON object:

```json
{
  "timestamp": "2026-04-03T10:30:01Z",
  "agent_role": "extractor",
  "phase": "extraction",
  "category": "commercial_lease_agreement",
  "prompt": "Extract the following...",
  "response": "{\"landlord_name\": \"Acme Properties LLC\", ...}",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "token_usage": { "prompt_tokens": 1240, "completion_tokens": 380 },
  "quality_tier": "high",
  "document_id": "doc_042"
}
```

### 6.3 Question Store Format

```json
{
  "category": "commercial_lease_agreement",
  "version": 2,
  "updated_at": "2026-04-03T10:00:00Z",
  "questions": [
    {
      "id": "q_001",
      "text": "Who is the landlord in this agreement?",
      "target_field": "landlord_name",
      "retrieval_priority": 1
    },
    {
      "id": "q_002",
      "text": "What is the monthly rental amount?",
      "target_field": "monthly_rent",
      "retrieval_priority": 1
    }
  ]
}
```

### 6.4 Prompt Population Format

```json
{
  "candidate_id": "candidate_001",
  "generation": 3,
  "created_at": "2026-04-03T02:00:00Z",
  "instructions": "You are a document extraction specialist. Given the retrieved pages from a commercial lease agreement, extract the following fields precisely...",
  "few_shot_ids": ["gs_001", "gs_003", "gs_007"],
  "fitness_scores": {
    "overall_accuracy": 0.87,
    "low_recovery_rate": 0.72,
    "high_preservation_rate": 1.0
  },
  "parent_ids": ["candidate_000"],
  "mutation_log": "Added explicit instruction to check for 'per annum' vs 'per month' rental distinctions."
}
```

---

## 7. Development Setup

### 7.1 Prerequisites

- **Python** 3.11+
- **uv** (recommended package manager) or pip
- **Git**
- At least one LLM provider API key (or a local model server running)

### 7.2 Installation

```bash
# Clone the repository
git clone <repo-url>
cd Meta-learning-for-document-extraction

# Create virtual environment
uv venv
# or: python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Unix:
source .venv/bin/activate

# Install dependencies
uv pip install -r docs/research/requirements.txt
# or: pip install -r docs/research/requirements.txt

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys
```

### 7.3 CLI Scripts

```bash
# Bootstrap a new category
python scripts/bootstrap_category.py --config configs/categories/commercial_lease.json

# Run Scout Agent for a category
python scripts/run_scout.py --category commercial_lease_agreement

# Run extraction on a document
python scripts/run_extraction.py --document ./samples/lease_042.pdf --category commercial_lease_agreement

# Trigger GEPA optimization
python scripts/run_optimization.py --category commercial_lease_agreement

# Export traces for SLM training
python scripts/export_traces.py --category commercial_lease_agreement --output ./exports/
```

### 7.4 Development Workflow

```
1. Feature branch from `main`
2. Write/update relevant schema in `src/schemas/`
3. Implement logic in the appropriate module
4. Add unit tests in `tests/unit/`
5. Add integration test if touching the graph or multi-module flow
6. Run full test suite: pytest tests/
7. Open PR with architecture doc reference if adding new components
```

---

## 8. Testing Strategy

| Layer | Scope | Approach |
|:---|:---|:---|
| **Unit** | Individual functions, schema validation, config loading, path resolution | Mock all LLM calls. Use fixture documents. Fast, deterministic. |
| **Integration** | Multi-module pipelines (Scout → Gold Standard, Input → Extract → Judge) | Mock LLM responses but test real file I/O, real retrieval indexing. |
| **Evaluation** | Extraction quality across a curated test corpus | Use Gold Standards as ground truth. Measure accuracy, precision, recall per field. |
| **Optimization** | GEPA improvement over baseline | Compare pre/post optimization metrics. Ensure no regressions on high-quality cases. |

**Key fixtures** (`tests/fixtures/`):
- Sample PDF and text documents per category
- Pre-built Gold Standard JSON files
- Mock LLM response payloads for deterministic testing

---

## 9. Deployment & Operations

### 9.1 Operational Modes

| Mode | Description | Trigger |
|:---|:---|:---|
| **Bootstrap** | Initialize a new category with schema + sample doc | Manual (CLI) |
| **Scout** | Run knowledge base construction for a category | Periodic / manual |
| **Production** | Process incoming documents for extraction | API call / batch |
| **Optimization** | Run GEPA cycle on accumulated evaluations | Nightly cron / manual |
| **Export** | Package trace logs for SLM fine-tuning | On-demand |

### 9.2 Monitoring

- **Trace logs** — every LLM interaction is logged with full prompt/response and metadata.
- **Quality distribution** — track the ratio of low/medium/high evaluations per category over time. A healthy system trends toward mostly `high`.
- **Optimization history** — track GEPA generation number, fitness scores, and deployed prompt versions per category.
- **Token usage** — aggregated from trace logs. Monitor cost per extraction, per category.

### 9.3 Scaling Considerations

| Concern | Mitigation |
|:---|:---|
| Large document volumes | Batch processing with async graph execution |
| Index size growth | Per-category indexes; prune stale indexes when Gold Standards are superseded |
| Trace storage | JSONL is append-only and compressible; rotate and archive periodically |
| Multi-category | Categories are fully isolated — separate configs, indexes, gold standards, prompts |

---

**Recommended implementation order:** `src/config/` (including `lm.py`) → `src/schemas/` → `src/storage/` → `src/retrieval/` → `src/agents/` → `src/orchestration/` → `src/optimization/`
