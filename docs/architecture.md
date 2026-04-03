***

# Architecture: Autonomous, Self-Optimizing Document Extraction System

## 1. Executive Summary
Current Document AI pipelines rely on static, human-engineered routing and brittle prompt engineering. This architecture proposes a shift from **stateless workflows** to a **stateful, continuous-learning system**. By combining multimodal late-interaction retrieval (ColPali/ColBERT), graph-based orchestration (LangGraph), and algorithmic prompt optimization (DSPy), the system autonomously routes documents, extracts data, evaluates its own performance, and updates its instructions to improve over time.

The system is designed to be **LLM-agnostic**. All agent components (Extraction, Judge, Scout) interface with language models through a configurable provider abstraction, allowing operators to swap between any compatible LLM (OpenAI, Anthropic, Gemini, local models, etc.) without modifying pipeline logic.

---

## 2. Core Architecture Components

### A. The Orchestration Layer (LangGraph)
LangGraph acts as the central state machine. It manages the flow of the document through various specialized agents, handles the Human-in-the-Loop (HitL) breakpoints, and maintains the state (document embeddings, draft JSON, judge scores) throughout the lifecycle.

### B. The Scout Layer (Knowledge Base Construction)
The Scout Agent is an offline, periodic component — not part of the live extraction path. Implemented as a DSPy RLM (Recursive Language Model) agent, the Scout leverages DSPy's built-in RLM capabilities — including its sandboxed REPL loop and code-generation tools — to parse extremely large datasets with the utmost accuracy. It acts as a knowledge base builder, inferring the core **questions** that need to be answered for extraction. These questions are critical: they become the actual **retrieval queries** fed into ColPali and ColBERT at extraction time, ensuring the Extraction Agent only ever sees the relevant pages or chunks — never the full document. If the input is a PDF, its first exploration processes the document as an image to analyze its visual topography.

### C. The Retrieval Layer (ColPali & ColBERT)
The system is built to be independent and malleable to both text and PDF inputs. Rather than relying heavily on traditional OCR and dense embeddings, the system uses late-interaction models to preserve granular semantic and spatial relationships. The Scout Agent's inferred questions are used as the retrieval queries for both routes.
* **Vision Route (ColPali):** For PDF inputs. It embeds document pages directly as visual patches, bypassing OCR. The Scout's questions retrieve only the relevant pages from the document, so extraction operates on targeted visual context rather than scanning entire documents.
* **Text Route (ColBERT):** For text-dense, narrative documents or direct text inputs. It calculates similarity at the token level using the MaxSim operation to preserve deep semantic nuance. The Scout's questions retrieve only the relevant chunks, narrowing extraction to precisely the text that matters:
    $$S(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} (q_i \cdot d_j)$$

### D. The Continuous Learning Layer (DSPy & GEPA)
DSPy replaces manual prompt engineering with prompt *programming*. It treats the extraction pipeline as code that can be compiled and optimized.
* **The Judge Agent:** A comparison function, not an independent evaluator. It compares the Extraction Agent's output directly against the Gold Standard produced by the Scout Agent (RLM phase). It only runs for document types that have Gold Standards — if no Gold Standard exists for a category, the Judge has nothing to compare against and does not evaluate. It assesses extraction quality as **low**, **medium**, or **high** and provides **textual feedback** identifying the specific differences between the extraction and the Gold Standard (e.g., missing fields, incorrect values, misinterpreted data).
* **GEPA (Genetic-Pareto Optimizer):** The core optimization engine. GEPA is a reflective, evolutionary optimizer that consumes the Judge's quality assessments and textual feedback to intelligently evolve extraction prompts. Low and medium-quality extractions are fed into GEPA for optimization; high-quality extractions confirm the current prompt is working. It operates by:
    1. **Capturing execution traces** of failed or sub-optimal extractions.
    2. **Reflective analysis** — a reflection LLM analyzes *why* each failure occurred by comparing the extraction trace against the Gold Standard.
    3. **Intelligent mutation** — instead of random prompt changes, GEPA proposes targeted instruction revisions informed by the diagnosed failure patterns.
    4. **Pareto-based selection** — maintaining a diverse population of prompt candidates that excel across different document edge cases, preventing collapse into a single suboptimal solution.
    5. **Component-level granularity** — because DSPy programs are modular, GEPA can optimize individual predictors (e.g., the extraction prompt) independently without disrupting other pipeline components.
* **Sample efficiency:** GEPA is effective with small training sets (30–100 Gold Standard examples), making it practical even during early system adoption when few approved extractions exist.

### E. The LLM Provider Layer (Agnostic)
All agents communicate with language models through a unified provider interface. The system supports any LLM backend — cloud-hosted APIs (OpenAI, Anthropic, Google, etc.) or locally-served models — configured via environment variables or JSON config. This allows operators to swap providers, adjust model sizes, or mix models per agent role (e.g., a smaller model for the Judge, a larger one for the Scout) without touching pipeline code.

### F. The Persistent Storage Layer (File System & JSON Configs)
To maintain a historical record for continuous learning, both the original source document and its corresponding **Gold Standard** (the approved, ideal extraction output) must be permanently stored so they can be referenced at any point in time. To prioritize simplicity and rapid iteration, this project will rely primarily on straightforward File System (FS) stores and JSON configuration files. There is no need to overcomplicate the architecture with complex databases at this stage.

### G. The Trace Logging Layer (SLM Training Pipeline)
Every LLM interaction across all phases of the system — Scout exploration, extraction, Judge evaluation, and GEPA optimization — is captured and persisted as a structured trace log. Each trace entry records the full prompt sent to the LLM and its complete response, along with metadata (timestamp, agent role, document category, phase, quality tier if applicable). These traces are stored in the FS alongside the Gold Standards.

The primary purpose is **future distillation**: once the system accumulates sufficient high-quality prompt-response pairs, this corpus becomes a training dataset for fine-tuning smaller, specialized language models (SLMs) to replicate the pipeline's behavior at a fraction of the inference cost. This positions the architecture for a natural evolution from LLM-powered agents to purpose-built SLMs as the system matures.

---

## 3. System Workflows

### Phase 1: Bootstrapping (Zero-Shot to Ground Truth)
To avoid manual data entry, new document categories are initialized using a lean payload and a Human-in-the-Loop approval process.

**1. The Lean Setup Payload:**
A user defines the schema and provides a sample document without providing the expected answers.
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
  "sample_document_uri": "./samples/lease_001.pdf"
}
```

**2. Zero-Shot Run & Human Approval:**
LangGraph routes the sample document to the Extractor Agent using only the base instructions. The system pauses and presents the draft JSON to a human. The human corrects any mistakes and hits "Approve."

**3. Promotion to Memory:**
The approved JSON is permanently stored in the FS store alongside the document's visual embedding and the original source document itself. These persisted files become the initial **Gold Standards** — the ground truth used for future few-shot prompting and Judge calibration. The Scout Agent (Phase 1.5) may later refine and supersede these with deeper, more rigorous Gold Standards.

### Phase 1.5: Knowledge Base & Gold Standard Construction (The Scout Phase)
Rather than executing on every single document passed to the system, the Scout Agent runs periodically or on-demand per document *type*.
1.  **Iterative Exploration:** It uses DSPy's RLM REPL loop to thoroughly explore sample documents of a specific category to unparalleled depths.
2.  **Question Inference & Context Maintenance:** It infers the essential questions that must be answered for extraction, dynamically populating an expanding knowledge base. These questions are then used as the **retrieval queries** for ColPali (to fetch relevant pages) and ColBERT (to fetch relevant chunks) during production extraction.
3.  **Gold Standard Construction:** Because of its rigorous reasoning capabilities, the Scout Agent builds the ideal extraction object — the definitive **Gold Standard** for the document type. This supersedes any initial Gold Standard from Phase 1 and is stored separately for extraction prompt optimization.

### Phase 2: Production Execution
When a new, unseen document or text payload enters the system:
1.  **Routing:** The system inspects the input type to determine the retrieval path. If the input is a PDF, it is routed to ColPali. If the input is raw text or a text-heavy format, it is routed to ColBERT. This is a straightforward input-type check, not a classification model.
2.  **Question-Driven Retrieval:** The Scout Agent's pre-inferred questions are used as queries against the chosen retrieval model. ColPali returns only the relevant **pages**; ColBERT returns only the relevant **chunks**. The Extraction Agent never processes the full document.
3.  **Context-Aware Extraction:** Working only with the retrieved pages/chunks, the Extraction Agent pulls the required data using the optimized instructions and few-shot examples compiled by DSPy.

### Phase 3: The Optimized Self-Healing Loop (GEPA-Driven)
The Gold Standards produced by the Scout Agent are the fuel for GEPA's optimization cycle. This is where the system transitions from "working" to "continuously improving":
1.  **Evaluation:** For document types with Gold Standards, the Judge Agent compares production extractions directly against the Gold Standard and assigns a quality tier — **low**, **medium**, or **high** — along with textual feedback identifying exactly where the extraction diverged.
2.  **Reflective Diagnosis (Offline):** Periodically (e.g., nightly batch runs), GEPA ingests the **low and medium** extraction traces alongside the Gold Standards. Its reflection LLM analyzes the gap between what was extracted and what the Gold Standard says *should* have been extracted, diagnosing root causes at the instruction level.
3.  **Evolutionary Optimization:** Using the diagnosed failures, GEPA proposes targeted prompt mutations — not random rewrites, but specific instruction changes that address the identified failure patterns. It maintains a population of candidate prompt versions and uses Pareto selection to keep the best-performing, most diverse set.
4.  **Few-Shot Synthesis:** Alongside prompt evolution, GEPA synthesizes superior few-shot examples drawn from the Gold Standard memory, ensuring the Extraction Agent has the most relevant examples for each document category.
5.  **Validation:** The optimized prompt is tested by re-running extraction on a **representative mix of low, medium, and high** examples. The Judge Agent re-evaluates these new extractions against the Gold Standards to verify the optimized prompt improves low/medium cases without degrading high-quality ones.
6.  **Deployment:** Once validated, the winning prompt variant replaces the old version, and the newly "compiled" Extractor Agent is deployed. The pipeline continuously self-improves with each optimization cycle.

---

## 4. Technology Stack Summary

| Component | Technology / Framework | Purpose |
| :--- | :--- | :--- |
| **State Orchestration** | LangGraph | Managing the agentic graph, routing, and Human-in-the-Loop breakpoints. |
| **LLM Provider** | Agnostic (configurable) | Swappable language model backend; any OpenAI-compatible or local model. |
| **Retrieval & Tools** | LlamaIndex / Custom | Handling document chunking, indexing, and executing tool calls. |
| **Visual Retrieval** | ColPali | Bypassing OCR for layout-aware, patch-level document understanding. |
| **Text Retrieval** | ColBERT | Token-level semantic search for narrative documents. |
| **Continuous Learning** | DSPy | Replacing prompt engineering with algorithmic metric-driven optimization. |
| **Data Validation** | Pydantic | Enforcing strict JSON schemas for all Extractor Agent outputs. |
| **Storage Layer** | FS & JSON Configs | Permanent file-system storage for Gold Standards; avoiding complex DBs. |
| **Trace Logging** | FS (structured logs) | Capturing all LLM prompt-response pairs for future SLM fine-tuning. |

***