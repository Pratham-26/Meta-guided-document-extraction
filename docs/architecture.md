***

# Architecture: Iteratively Building the Best Self-Improving Document Extraction System

## 1. Executive Summary

**Project philosophy:** We are iteratively building the best self-improving document extraction system using agents. Every design decision, every module, and every optimization cycle exists to push the system closer to that goal. The architecture is not a fixed blueprint — it is a living specification that evolves as we learn what works.

Current Document AI pipelines rely on static, human-engineered routing and brittle prompt engineering. This architecture proposes a shift from **stateless workflows** to a **stateful, continuous-learning system**. By combining multimodal late-interaction retrieval (ColPali/ColBERT), graph-based orchestration (LangGraph), and algorithmic prompt optimization (DSPy), the system autonomously routes documents, extracts data, evaluates its own performance, and updates its instructions to improve over time.

The system is designed to be **LLM-agnostic**. All agent components (Extraction, Judge, Scout) interface with language models through a configurable provider abstraction, allowing operators to swap between any compatible LLM (OpenAI, Anthropic, Gemini, local models, etc.) without modifying pipeline logic.

---

## 2. Context Identity: The (Category, Input Modality) Key

Every pipeline customization — inferred questions, Gold Standards, optimized prompts, retrieval indexes, and trace logs — is scoped to the **composite key of `(category, input_modality)`**, not just the category alone. A `commercial_lease_agreement` submitted as a PDF and the same category submitted as raw text maintain **completely independent contexts**. They have separate bootstrapping flows, separate questions, separate Gold Standards, separate GEPA optimization cycles, and separate retrieval indexes. This is because the retrieval strategy (ColPali vs ColBERT), the extraction prompts, and the visual vs textual patterns are fundamentally different depending on input modality. Optimizing for one does not imply optimization for the other.

The input modality is determined at ingestion time by file type (PDF → `pdf`, text/text-heavy formats → `text`) and is immutable for the lifetime of a given context. The storage layout, context gate, bootstrapping, evaluation, and optimization all operate on this composite key.

---

## 3. Core Architecture Components

### A. The Orchestration Layer (LangGraph)

The graph carries both `category_name` and `input_modality` through its state. Every node that resolves context — questions, Gold Standards, prompts, indexes — uses both values to locate the correct artifact set.
LangGraph acts as the central state machine. It manages the flow of the document through various specialized agents, handles conditional routing (gold document detection, modality-based retrieval), and maintains the state (document embeddings, draft JSON, judge scores) throughout the lifecycle.

### B. The Scout Layer (Knowledge Base Construction)
The Scout Agent is an offline, periodic component — not part of the live extraction path. Implemented primarily as a DSPy RLM (Recursive Language Model) agent, the Scout leverages DSPy's built-in RLM capabilities — including its sandboxed REPL loop and code-generation tools — to parse extremely large datasets with the utmost accuracy. Exploration uses `dspy.RLM` for deep reasoning, while question inference uses `dspy.Predict` for efficient structured output. It acts as a knowledge base builder, inferring the core **questions** that need to be answered for extraction. These questions are critical: they become the actual **retrieval queries** fed into ColPali and ColBERT at extraction time, ensuring the Extraction Agent only ever sees the relevant pages or chunks — never the full document. If the input is a PDF, its first exploration processes the document as an image to analyze its visual topography.

**Modality scoping:** The Scout produces questions and Gold Standards for a specific `(category, input_modality)` pair. A category needs separate bootstrapping for PDF and text modalities if both are expected in production.

### C. The Retrieval Layer (ColPali & ColBERT)
The system is built to be independent and malleable to both text and PDF inputs. Rather than relying heavily on traditional OCR and dense embeddings, the system uses late-interaction models to preserve granular semantic and spatial relationships. The Scout Agent's inferred questions are used as the retrieval queries for both routes.
* **Vision Route (ColPali):** For PDF inputs. It embeds document pages directly as visual patches, bypassing OCR. The Scout's questions retrieve only the relevant pages from the document, so extraction operates on targeted visual context rather than scanning entire documents.
* **Text Route (ColBERT):** For text-dense, narrative documents or direct text inputs. The system parses structured text formats (`.docx`, `.html`/`.htm`, `.rtf`, `.odt`) using format-specific libraries, and falls back to raw UTF-8 reads for plain-text formats (`.txt`, `.md`, `.csv`, etc.). Extracted text is chunked and indexed by ColBERT, which calculates similarity at the token level using the MaxSim operation to preserve deep semantic nuance. The Scout's questions retrieve only the relevant chunks, narrowing extraction to precisely the text that matters:
    $$S(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} (q_i \cdot d_j)$$

### D. The Continuous Learning Layer (DSPy & GEPA)
DSPy replaces manual prompt engineering with prompt *programming*. It treats the extraction pipeline as code that can be compiled and optimized.
* **The Judge Agent:** A comparison function, not an independent evaluator. It compares the Extraction Agent's output directly against the Gold Standard produced by the Scout Agent. It only runs for **gold documents** — documents designated as gold via user flag (`--gold`) or random sampling. For regular (non-gold) documents, the Judge is skipped entirely and the extraction result is returned directly. Gold Standards accumulate from three sources: bootstrapping (initial sample documents), auto-gold (first N documents per category/modality automatically treated as gold — default 10, configurable via `auto_gold_initial_count` in `process_config.json`), user-flagged documents (`--gold`), and random sampling (configurable rate per category/modality, kicks in only after the auto-gold phase). The Judge assesses extraction quality as **low**, **medium**, or **high** and provides **textual feedback** identifying the specific differences between the extraction and the Gold Standard (e.g., missing fields, incorrect values, misinterpreted data).
* **GEPA (Genetic-Pareto Optimizer):** The core optimization engine. GEPA is a reflective, evolutionary optimizer that consumes the Judge's quality assessments and textual feedback to intelligently evolve extraction prompts. Low and medium-quality extractions are fed into GEPA for optimization; high-quality extractions confirm the current prompt is working. It operates by:
    1. **Capturing execution traces** of failed or sub-optimal extractions.
    2. **Reflective analysis** — a reflection LLM analyzes *why* each failure occurred by comparing the extraction trace against the Gold Standard.
    3. **Intelligent mutation** — instead of random prompt changes, GEPA proposes targeted instruction revisions informed by the diagnosed failure patterns.
    4. **Pareto-based selection** — maintaining a diverse population of prompt candidates that excel across different document edge cases, preventing collapse into a single suboptimal solution.
    5. **Component-level granularity** — because DSPy programs are modular, GEPA can optimize individual predictors (e.g., the extraction prompt) independently without disrupting other pipeline components.
* **Sample efficiency:** GEPA is effective with small training sets (30–100 Gold Standard examples), making it practical even during early system adoption when few approved extractions exist.

### E. The LLM Provider Layer (DSPy + LiteLLM)
All agents communicate with language models through **DSPy's built-in LiteLLM integration**. Because DSPy uses LiteLLM internally as its LLM router, the system inherits support for 100+ providers (OpenAI, Anthropic, Google, Azure, Ollama, vLLM, HuggingFace, etc.) out of the box — no custom provider abstraction needed. Each agent role is mapped to one or more LiteLLM model strings in a JSON config file, and API keys are resolved automatically from standard environment variables.

**Dual-model roles:** The Scout and Extractor agents handle both PDF documents (processed as images) and text documents. Each can be configured with separate `text_model` and `vision_model` entries. The system selects the correct model at runtime based on input type — PDFs use the vision model (e.g., `zai/glm-4.6v-flash`), text files use the text model (e.g., `zai/glm-4.7-flash`). This allows operators to use cheaper or faster text models for text-heavy workloads while using dedicated vision models for PDF documents with complex layouts, tables, and spatial relationships.

**Single-model roles:** The Judge and Reflector agents only process text (extraction results, prompts, evaluations) and use a single `model` entry.

This allows operators to swap providers, adjust model sizes, or mix models per agent role (e.g., a vision model for the Scout on PDFs, a text model on text files, a smaller model for the Judge) with a one-line config change and zero code modifications.

### F. The Persistent Storage Layer (File System & JSON Configs)
To maintain a historical record for continuous learning, both the original source document and its corresponding **Gold Standard** (the approved, ideal extraction output) must be permanently stored so they can be referenced at any point in time. To prioritize simplicity and rapid iteration, this project will rely primarily on straightforward File System (FS) stores and JSON configuration files. There is no need to overcomplicate the architecture with complex databases at this stage.

### G. The Trace Logging Layer (SLM Training Pipeline)
Every LLM interaction across all phases of the system — Scout exploration, extraction, Judge evaluation, and GEPA optimization — is captured and persisted as a structured trace log. Each trace entry records the full prompt sent to the LLM and its complete response, along with metadata (timestamp, agent role, document category, phase, quality tier if applicable). These traces are stored in the FS alongside the Gold Standards.

The primary purpose is **future distillation**: once the system accumulates sufficient high-quality prompt-response pairs, this corpus becomes a training dataset for fine-tuning smaller, specialized language models (SLMs) to replicate the pipeline's behavior at a fraction of the inference cost. This positions the architecture for a natural evolution from LLM-powered agents to purpose-built SLMs as the system matures.

---

## 3. System Workflows

### Phase 1: Bootstrapping (Scout-Driven Context Creation)
When a new `(category, input_modality)` combination has **no existing context** (no Gold Standards, no inferred questions for that modality), the system cannot perform question-driven retrieval or Judge evaluation. Bootstrapping solves this by requiring the Scout Agent to analyze **two sample documents of the matching input type** to build the initial knowledge base before any production extraction occurs.

**1. The Lean Setup Payload:**
A user defines the schema, provides extraction instructions, and supplies **two sample documents** — no expected answers required. Two documents are the minimum because the Scout needs to cross-reference patterns across documents to distinguish category-level structure from document-specific noise.
```json
{
  "category_name": "commercial_lease_agreement",
  "expected_schema": {
    "type": "object",
    "properties": {
      "landlord_name": {"type": "string"},
      "monthly_rent": {"type": "number"}
    }
  },
  "extraction_instructions": "Extract the names of the signing parties and the financial terms.",
  "sample_documents": [
    "./samples/lease_001.pdf",
    "./samples/lease_002.pdf"
  ]
}
```

**2. Scout Agent Runs on Both Documents:**
The Scout Agent (DSPy RLM) processes both sample documents through its full exploratory loop:
1.  **Iterative Exploration:** It uses DSPy's RLM REPL loop to thoroughly explore each document to unparalleled depths. If the input is a PDF, its first pass processes the document as an image to analyze visual topography.
2.  **Question Inference:** By analyzing both documents, the Scout infers the essential **questions** that need to be answered for extraction of this category. Cross-referencing two documents ensures the questions target category-level patterns, not one-off quirks. These questions become the **retrieval queries** for ColPali and ColBERT during production extraction.
3.  **Gold Standard Construction:** For each document, the Scout builds the ideal extraction object — the definitive **Gold Standard** — using its rigorous reasoning capabilities.

**3. Promotion to Memory:**
The Gold Standards, inferred questions, and original source documents are automatically promoted to the knowledge base and permanently stored in the FS store under `data/categories/<name>/<modality>/`. The `(category, modality)` combination is now **context-ready** — production extraction, Judge evaluation, and GEPA optimization can all operate for that specific modality.

### Phase 1.5: Ongoing Knowledge Base Refinement (The Scout Phase)
After bootstrapping, the Scout Agent continues to run **periodically or on-demand** per document category to deepen and refine the knowledge base.
1.  **Expanded Exploration:** As more documents of the category are encountered, the Scout can be re-run on additional samples to discover new patterns, edge cases, or structural variations.
2.  **Question Refinement:** The question set evolves — new questions are added, irrelevant ones pruned — as the Scout's understanding of the category deepens.
3.  **Gold Standard Expansion:** New Gold Standards are added to the store, giving GEPA more training signal and the Judge more comparison points. These supersede earlier Gold Standards when they represent higher-quality extractions.

### Phase 2: Production Execution
When a new, unseen document or text payload enters the system:
1.  **Context Gate:** The system checks whether Scout context (questions + Gold Standards) exists for the document's `(category, input_modality)` combination. If no context exists, the document is **queued** and the operator is prompted to run the bootstrapping flow (Phase 1) with two sample documents of the matching modality before extraction can proceed.
2.  **Gold Detection:** The system determines whether this is a **gold document** — because it was auto-flagged (the first N documents per category/modality are automatically gold to build an initial set for GEPA), because the user flagged it (`--gold` flag), or because it was selected via random sampling (configurable per `(category, modality)` pair). The auto-gold threshold is configured in `process_config.json` (`auto_gold_initial_count`, default 10). User-flagged and auto-gold documents count toward the total. Gold documents trigger the Scout Agent to build a Gold Standard and merge new questions into the existing question set before extraction proceeds. Regular documents skip the Scout entirely.
3. **Routing:** The system inspects the input type to determine the retrieval path. If the input is a PDF, it is routed to ColPali. If the input is raw text or a text-based format (`.docx`, `.html`/`.htm`, `.rtf`, `.odt`, `.txt`, `.md`, `.csv`, etc.), it is parsed into plain text and routed to ColBERT. This is a straightforward input-type check based on file extension, not a classification model.
4.  **Question-Driven Retrieval:** The Scout Agent's pre-inferred questions are used as queries against the chosen retrieval model. ColPali returns only the relevant **pages**; ColBERT returns only the relevant **chunks**. The Extraction Agent never processes the full document.
5.  **Context-Aware Extraction:** Working only with the retrieved pages/chunks, the Extraction Agent pulls the required data using the optimized instructions and few-shot examples compiled by DSPy.
6.  **Judge Evaluation (Gold Documents Only):** For gold documents, the Judge Agent compares the extraction against the Gold Standard produced by the Scout and assigns a quality tier with textual feedback. Regular documents skip the Judge entirely and return the extraction result directly.

### Phase 3: The Optimized Self-Healing Loop (GEPA-Driven)
The Gold Standards produced by the Scout Agent are the fuel for GEPA's optimization cycle. This is where the system transitions from "working" to "continuously improving":
1.  **Evaluation:** For gold documents only, the Judge Agent compares the extraction against the Gold Standard and assigns a quality tier — **low**, **medium**, or **high** — along with textual feedback identifying exactly where the extraction diverged.
2.  **Reflective Diagnosis (Offline):** Periodically (e.g., nightly batch runs), GEPA ingests the **low and medium** extraction traces alongside the Gold Standards. Its reflection LLM analyzes the gap between what was extracted and what the Gold Standard says *should* have been extracted, diagnosing root causes at the instruction level.
3.  **Evolutionary Optimization:** Using the diagnosed failures, GEPA proposes targeted prompt mutations — not random rewrites, but specific instruction changes that address the identified failure patterns. It maintains a population of candidate prompt versions and uses Pareto selection to keep the best-performing, most diverse set.
4.  **Few-Shot Synthesis:** Alongside prompt evolution, GEPA synthesizes superior few-shot examples drawn from the Gold Standard memory, ensuring the Extraction Agent has the most relevant examples for each document category.
5. **Validation:** The optimized prompt is tested by re-running extraction on a sample of Gold Standards. The Judge Agent re-evaluates these new extractions against the Gold Standards to verify the optimized prompt improves quality.
6.  **Deployment:** Once validated, the winning prompt variant replaces the old version, and the newly "compiled" Extractor Agent is deployed. The pipeline continuously self-improves with each optimization cycle.

---

## 4. Technology Stack Summary

| Component | Technology / Framework | Purpose |
| :--- | :--- | :--- |
| **State Orchestration** | LangGraph | Managing the agentic graph, conditional routing, and gold document detection. |
| **LLM Provider** | DSPy + LiteLLM | Built-in multi-provider LLM routing; 100+ providers via unified model strings. |
| **Retrieval & Tools** | LlamaIndex / Custom | Handling document chunking, indexing, and executing tool calls. |
| **Visual Retrieval** | ColPali | Bypassing OCR for layout-aware, patch-level document understanding. |
| **Text Retrieval** | ColBERT | Token-level semantic search for narrative documents. |
| **Text Parsing** | python-docx, BeautifulSoup, odfpy, striprtf | Extracting plain text from structured formats (DOCX, HTML, RTF, ODT). |
| **Continuous Learning** | DSPy | Replacing prompt engineering with algorithmic metric-driven optimization. |
| **Data Validation** | Pydantic | Enforcing strict JSON schemas for all Extractor Agent outputs. |
| **Storage Layer** | FS & JSON Configs | Permanent file-system storage for Gold Standards; avoiding complex DBs. |
| **Trace Logging** | FS (structured logs) | Capturing all LLM prompt-response pairs for future SLM fine-tuning. |

***