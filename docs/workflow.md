# Workflow

This document describes the end-to-end workflow for extracting data from documents using this system. Everything is scoped by the composite key `(category, input_modality)` -- a PDF and a text file of the same category are treated as completely separate contexts.

---

## Step 0: Bootstrap a New Category (One-Time Setup)

Before any document can be extracted, the system needs a knowledge base for that category. This only needs to happen once per `(category, modality)` pair.

1. **Create a category config** (`configs/categories/<name>.json`) with:
   - `category_name` -- the document type (e.g. `commercial_lease_agreement`)
   - `expected_schema` -- the JSON schema of what to extract
   - `extraction_instructions` -- natural language guidance for extraction
   - `sample_documents` -- at least 2 sample files of the matching modality

2. **Run the bootstrap script:**
   ```bash
   python scripts/bootstrap_category.py --config configs/categories/commercial_lease.json
   ```
   This validates the config, saves it, creates the directory structure under `data/categories/<name>/`, and copies the sample documents into storage.

3. **Run the Scout Agent** to build the knowledge base:
   ```bash
   python scripts/run_scout.py --category commercial_lease_agreement
   ```
   The Scout:
   - Explores each sample document (PDFs are processed as images for visual topography)
   - Cross-references both documents to distinguish category-level patterns from one-off quirks
   - Builds a Gold Standard (ideal extraction) for each document
   - Infers retrieval questions that will be used as queries during production extraction
   - Saves everything to `data/categories/<name>/<modality>/`

   After this step, the category is **context-ready** -- production extraction can proceed.

---

## Step 1: User Submits a Document for Extraction

The user provides:
- **A document** (PDF or text file)
- **A category name** (must match an already-bootstrapped category)
- Optionally, a **`--gold` flag** to designate this document as a Gold Standard candidate

```bash
# Regular extraction
python scripts/run_extraction.py --document ./incoming/lease_042.pdf --category commercial_lease_agreement

# User-flagged gold document
python scripts/run_extraction.py --document ./incoming/lease_042.pdf --category commercial_lease_agreement --gold
```

The system automatically determines the input modality from the file extension (`.pdf` -> `pdf`, everything else -> `text`).

---

## Step 2: Context Gate

The system checks whether Scout context (Gold Standards + inferred questions) exists for the `(category, modality)` pair.

- **Context exists** -> continues to Step 3
- **No context** -> halts with an error telling the operator to run bootstrapping first

---

## Step 3: Load Category Config

The category config is loaded from `configs/categories/<name>.json`, giving the system the expected output schema and extraction instructions.

---

## Step 4: Gold Detection

The system determines whether this document is a **gold document**. There are three sources of gold documents:

| Source | Trigger |
|:---|:---|
| **User flag** | `--gold` flag on the CLI |
| **Random sampling** | Every Nth document (configurable per category/modality) |

Random sampling works via a persistent counter per `(category, modality)`. The sampling rate defaults to 1 in 100 (set in `configs/process_config.json`) and can be overridden per combination in `data/categories/<name>/<modality>/sampling_config.json`.

- If the document is gold -> **gold path** (Step 5a then Step 6)
- If the document is not gold -> **regular path** (Step 5b then Step 6)

---

## Step 5a: Gold Path -- Scout Processing

For gold documents, the Scout Agent runs on the document before extraction:

1. **Scout explores the document** -- deep analysis of structure and content (image-first pass for PDFs)
2. **Builds a Gold Standard** -- the ideal extraction for this document, saved to `data/categories/<name>/<modality>/gold_standards/`
3. **Infers retrieval questions** -- determines what questions need to be answered for this document type
4. **Merges questions** -- new questions are compared against the existing question set. Questions targeting fields not yet covered are added; existing fields are left unchanged. The question set version is incremented.

---

## Step 5b: Regular Path -- Skip Scout

For regular documents, the Scout does not run. The pipeline proceeds directly to retrieval using the existing question set.

---

## Step 6: Shared Extraction Pipeline

Both gold and regular documents go through the same extraction pipeline:

1. **Load questions** -- the Scout's inferred questions are loaded from `data/categories/<name>/<modality>/questions/questions.json`
2. **Route by input modality:**

   | Input Type | Retrieval Engine | What It Returns |
   |:---|:---|:---|
   | PDF | **ColPali** (visual) | Relevant page images |
   | Text | **ColBERT** (token-level) | Relevant text chunks |

3. **Question-driven retrieval** -- the Scout's questions are used as queries to retrieve only the relevant pages or chunks. The Extraction Agent never sees the full document.
4. **Extraction** -- the Extractor Agent produces a structured JSON validated against the category's schema, using compiled instructions and few-shot examples.

---

## Step 7: Judge Evaluation (Gold Documents Only)

The Judge Agent only runs for gold documents. It compares the extraction output against the Scout's Gold Standard for this document:

- **Quality tier:** `low`, `medium`, or `high`
- **Feedback:** textual explanation of where the extraction diverged
- **Field-level diffs:** per-field comparison details

For regular documents, the Judge is skipped entirely and the extraction result is returned directly.

---

## Step 8: Trace Logging

All LLM interactions (extraction prompts/responses, Judge evaluations) are logged as structured JSONL traces to `data/traces/<category>/<modality>/`. These traces serve two purposes:

1. **Debugging and monitoring** -- full audit trail of every decision
2. **Future SLM distillation** -- accumulated traces become a training corpus for fine-tuning smaller, specialized models

---

## Step 9: Self-Improvement via GEPA (Periodic / On-Demand)

After enough gold document evaluations accumulate (especially low/medium quality ones), the GEPA optimizer runs to improve extraction prompts:

```bash
python scripts/run_optimization.py --category commercial_lease_agreement --modality pdf
```

The GEPA cycle:
1. **Collects** low and medium quality Judge evaluations (from gold documents only)
2. **Reflects** -- a reflection LLM diagnoses *why* each extraction failed by comparing traces against Gold Standards
3. **Mutates** -- proposes targeted prompt revisions based on diagnosed failure patterns
4. **Selects** -- Pareto-based selection maintains a diverse population of prompt candidates
5. **Validates** -- re-runs extraction on a mix of low/medium/high examples and re-evaluates
6. **Deploys** -- the winning prompt replaces `prompts/current.json`

This loop runs independently of production extraction (nightly batch or on-demand) and continuously improves the system over time.

---

## Gold Standard Sources -- Summary

Gold Standards are the system's ground truth. They accumulate from three sources:

| Source | When | How |
|:---|:---|:---|
| **Bootstrap** | During initial setup | Scout processes 2 sample docs, builds Gold Standards |
| **User flag** | Any time via `--gold` | Scout processes the doc, builds Gold Standard, Judge evaluates |
| **Random sampling** | Automatically every Nth doc | Same pipeline as user-flagged gold docs |

All gold documents go through the full Scout -> Extract -> Judge pipeline. Regular documents only go through Extract.

---

## Summary Flow

```
User submits document + category [--gold optional]
        |
        v
[Context Gate] -- no context --> HALT (bootstrap first)
        |
        | context exists
        v
[Load category config]
        |
        v
[Detect gold] -- user flag or random sample triggers gold path
        |
   +----+----+
   |         |
   v         v
 [Gold]    [Regular]
   |         |
   v         |
[Scout:     |
 Gold Std,  |
 Questions] |
   |         |
   +----+----+
        |
        v
[Route: PDF --> ColPali | Text --> ColBERT]
        |
        v
[Retrieve relevant pages/chunks]
        |
        v
[Extractor Agent: structured JSON]
        |
   +----+----+
   |         |
   v         v
 [Gold]    [Regular]
   |         |
   v         |
[Judge      |
 evaluates] |
   |         |
   +----+----+
        |
        v
[Log traces + return result]

--- Separate offline loop ---

[GEPA: collect failures --> diagnose --> mutate prompts --> validate --> deploy]
```
