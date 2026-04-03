Here is the consolidated architecture document, outlining the design, workflows, and tech stack for the self-optimizing Document AI system we discussed.

***

# Architecture: Autonomous, Self-Optimizing Document Extraction System

## 1. Executive Summary
Current Document AI pipelines rely on static, human-engineered routing and brittle prompt engineering. This architecture proposes a shift from **stateless workflows** to a **stateful, continuous-learning system**. By combining multimodal late-interaction retrieval (ColPali/ColBERT), graph-based orchestration (LangGraph), and algorithmic prompt optimization (DSPy), the system autonomously routes documents, extracts data, evaluates its own performance, and updates its instructions to improve over time.

---

## 2. Core Architecture Components

### A. The Orchestration Layer (LangGraph)
LangGraph acts as the central state machine. It manages the flow of the document through various specialized agents, handles the Human-in-the-Loop (HitL) breakpoints, and maintains the state (document embeddings, draft JSON, judge scores) throughout the lifecycle.

### B. The Retrieval & Extraction Layer
The system is built to be independent and malleable to both text and PDF inputs. Rather than relying heavily on traditional OCR and dense embeddings, the system uses late-interaction models to preserve granular semantic and spatial relationships.
* **The "Scout" Agent (Knowledge Base Builder):** Implemented as an RLM (Recursive Language Model) in DSPy to parse extremely large datasets with the utmost accuracy. Instead of loading full document contexts, the Scout Agent operates in an iterative REPL loop, receiving metadata and dynamically writing Python code to explore the data. It acts as a knowledge base builder, determining the core questions that need to be asked for extraction. It runs periodically or on-demand to maintain an expanding, context-rich knowledge base for the Extraction Agent. If the input is a PDF, its first exploration processes the document as an image to analyze its visual topography.
* **Vision Route (ColPali):** For documents relying heavily on layout (e.g., nested tables, complex forms). It embeds document pages directly as visual patches, bypassing OCR.
* **Text Route (ColBERT):** For text-dense, narrative documents or direct text inputs. It calculates similarity at the token level using the MaxSim operation to preserve deep semantic nuance:
    $$S(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} (q_i \cdot d_j)$$

### C. The Continuous Learning Layer (DSPy)
DSPy replaces manual prompt engineering with prompt *programming*. It treats the extraction pipeline as code that can be compiled and optimized.
* **The Judge Agent:** Evaluates the extracted JSON against schema validation, completeness, and hallucination checks, returning a numeric score.
* **The Optimizer:** If the Judge scores an extraction poorly, the DSPy optimizer automatically rewrites the Extraction Agent's instructions and selects new few-shot examples from the ground-truth memory to fix the edge case.

### D. The Persistent Storage Layer (File System & JSON Configs)
To maintain a historical record for continuous learning, any data source that has a "gold" baseline (or "gold response") generated must be permanently stored so it can be referenced at any point in time. To prioritize simplicity and rapid iteration, this project will rely primarily on straightforward File System (FS) stores and JSON configuration files. There is no need to overcomplicate the architecture with complex databases at this stage.

---

## 3. System Workflows

### Phase 1: Bootstrapping (Zero-Shot to Ground Truth)
To avoid manual data entry, new document categories are initialized using a lean payload and a Human-in-the-Loop approval process.

**1. The Lean Setup Payload:**
A user defines the schema and provides a sample document without providing the expected answers.
```json
{
  "category_name": "commercial_lease_agreement",
  "routing_preference": "colbert_text_heavy", 
  "expected_schema": {
    "type": "object",
    "properties": {
      "landlord_name": {"type": "string"},
      "monthly_rent": {"type": "number"}
    }
  },
  "extraction_instructions": "Extract the names of the signing parties and the financial terms.",
  "sample_document_uri": "s3://bucket/samples/lease_001.pdf"
}
```

**2. Zero-Shot Run & Human Approval:**
LangGraph routes the sample document to the Extractor Agent using only the base instructions. The system pauses and presents the draft JSON to a human. The human corrects any mistakes and hits "Approve."

**3. Promotion to Memory:**
The approved JSON is permanently stored in the FS store alongside the document's visual embedding. These persisted files become our "Golden Templates", forming the ground truth used for future few-shot prompting and Judge calibration.

### Phase 1.5: Knowledge Base & Golden Baseline (The Scout Phase)
Rather than executing on every single document passed to the system, the Scout Agent runs periodically or on-demand per document *type*.
1.  **Iterative Exploration:** It uses its RLM REPL loop to thoroughly explore sample documents of a specific category to unparalleled depths.
2.  **Question Inference & Context Maintenance:** It figures out what data needs to be extracted by inferring the essential questions that must be answered, dynamically populating an expanding knowledge base.
3.  **Golden Object Construction:** Because of its rigorous reasoning capabilities, the Scout Agent excels at building the ideal extraction object (the "gold response"). This perfect output serves as the undisputed ground truth for the document type and is stored separately to be later used for extraction prompt optimization.

### Phase 2: Production Execution (The Swarm)
When a new, unseen document or text payload enters the system:
1.  **Routing:** The system evaluates the input modality (text vs. PDF) and routes the document to the optimal retrieval path. ColPali is used for complex, visually-rich PDFs, while ColBERT handles text-dense or raw text payloads.
2.  **Context-Aware Extraction:** The Extraction Agent taps into the growable context established by the Scout Agent, pulling the required data using the optimized instructions and few-shot examples compiled by DSPy.

### Phase 3: The Optimized Self-Healing Loop
By harnessing the inferred questions, the Scout's meticulous gold responses, and the swift routing of ColPali/ColBERT, the system forms an extraction pipeline that autonomously self-heals:
1.  **Evaluation:** The Judge Agent scores the faster production extractions against the baselines and principles established by the Scout Agent.
2.  **Diagnosis & Optimization (Offline):** Periodically (e.g., nightly batch runs), DSPy gathers sub-optimal or human-corrected extractions. Crucially, it retrieves the separately stored gold responses and runs GEPA (built into DSPy) to autonomously recognize failures, optimize the extraction prompt, and synthesize superior few-shot examples.
3.  **Deployment:** The newly "compiled" Extractor Agent replaces the old version, allowing the pipeline to continuously self-improve and correct edge cases without manual developer intervention.

---

## 4. Technology Stack Summary

| Component | Technology / Framework | Purpose |
| :--- | :--- | :--- |
| **State Orchestration** | LangGraph | Managing the agentic graph, routing, and Human-in-the-Loop breakpoints. |
| **Retrieval & Tools** | LlamaIndex / Custom | Handling document chunking, indexing, and executing tool calls. |
| **Visual Retrieval** | ColPali | Bypassing OCR for layout-aware, patch-level document understanding. |
| **Text Retrieval** | ColBERT | Token-level semantic search for narrative documents. |
| **Continuous Learning** | DSPy | Replacing prompt engineering with algorithmic metric-driven optimization. |
| **Data Validation** | Pydantic | Enforcing strict JSON schemas for all Extractor Agent outputs. |
| **Storage Layer** | FS & JSON Configs | Permanent file-system storage for gold baselines; avoiding complex DBs. |

***