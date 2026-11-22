# Known Limitations and Architectural Trade-Offs

This document details the empirical findings, engineering trade-offs, and operational boundaries of the **Agentic AI-Based Academic Research and Learning Assistant**, based on automated benchmarking across Phase 1 through Phase 8.

---

## 1. Baseline vs. Agentic RAG: Factual Accuracy Parity

In the Phase 7b reproducible evaluation (30 questions across 3 distinct papers: *Attention Is All You Need*, *LeafScan*, and *GridTCN*), the comparative findings revealed:

| Metric | Baseline RAG (Single-Turn) | Agentic LangGraph RAG |
| :--- | :---: | :---: |
| **Overall Strict Correctness** | **83.3%** (25 / 30) | **83.3%** (25 / 30) |
| **Partial Factual Accuracy (Mean Score)** | **90.6%** (27.17 / 30.00) | **90.6%** (27.17 / 30.00) |
| **Answerable Strict Accuracy (Covered)** | **76.2%** (16 / 21) | **76.2%** (16 / 21) |
| **Refusal Recall (True Refusals)** | **100.0%** (9 / 9) | **100.0%** (9 / 9) |
| **Refusal Precision** | **90.0%** (9 / 10) | **90.0%** (9 / 10) |
| **Citation Reference Validity** | **100.0%** (35 / 35 supported) | **96.4%** (27 / 28 supported) |
| **Average End-to-End Latency** | **14.03s** (P50: 11.07s) | **22.37s** (P50: 18.96s; 1.59x baseline) |
| **Average LLM Calls per Query** | **1.0 call** (30 total) | **2.6 calls** (78 total; 2.60x baseline) |

### Why Agentic Orchestration Did Not Exceed Baseline Factual Accuracy
1. **Direct Fact Extraction vs. Multi-Stage Synthesis:** For direct academic questions (e.g., *"What learning rate warmup steps were used?"* or *"What was the test set BLEU score?"*), dense vector retrieval (`all-MiniLM-L6-v2`) with cosine similarity thresholding retrieves the exact relevant paragraphs with high precision. A single focused LLM call reproduces the factual answer accurately.
2. **Paraphrasing and Information Loss in Multi-Node Chains:** In the agentic pipeline, questions flow through the `router_node`, `qa_retrieve`, `qa_answer`, and `update_memory` nodes. While this enables multi-turn context retention, clarifying dialogues, and autonomous query rewriting, each intermediary LLM generation presents a small surface area for slight paraphrasing or omission of fine-grained numerical details.
3. **Architectural Role Distinction:** The agentic architecture's primary value is not micro-optimization of single-hop factual extraction, but rather **complex workflow orchestration**: dynamic routing between Q&A, structured note generation, slide deck extraction, multi-turn conversational memory, and autonomous unanswerable-query rejection.

---

## 2. Latency and Compute Cost Multipliers

- **Query-Level Multipliers:**
  - Agentic Q&A incurs a **~1.59x latency multiplier** (mean 22.37s vs. 14.03s; P50: 18.96s vs 11.07s) and a **2.60x LLM call multiplier** (mean 2.6 calls vs 1.0 call per turn, up to 4 calls during query rewrite retries).
  - On rate-limited cloud LLM endpoints (such as the Groq free tier with 30 RPM / 6,000 TPM for LLaMA 3.3 70B), concurrent agentic queries can saturate token allowances faster than single-turn systems.
- **Cold vs. Warm Artifact Generation:**
  - **Cold Synthesis:** Generating complete structured notes or an 8-slide presentation outline requires decomposing paper sections, multi-stage summarization, and bullet synthesis. This takes **15–35 seconds** end-to-end (or longer if API rate limits require exponential backoff retries).
  - **Warm Retrieval:** The system implements an idempotent SQLite artifact cache. Once generated, subsequent requests for summaries, notes, or outlines are served in **< 15 milliseconds** (a >1,000x speedup), completely bypassing the LLM.

---

## 3. Benchmark Dataset Scope and Statistical Generalizability

- **Dataset Scale:** The benchmark suite comprises **30 carefully curated questions** across 3 distinct peer-reviewed domains:
  - Natural Language Processing (*Attention Is All You Need*, Vaswani et al., 2017)
  - Computer Vision in Agriculture (*LeafScan Disease Detection*, 2024)
  - Time-Series Forecasting in Power Grids (*GridTCN Load Forecasting*, 2024)
- **Scope Boundary:** While 30 questions provide reproducible comparative metrics across varying chunk layouts and domain terminologies, it is not a 10,000-question industry benchmark like MS MARCO or MMLU. Complex edge cases—such as deeply nested multi-column tables, mathematical equation symbol parsing, and cross-document multi-hop reasoning spanning dozens of papers simultaneously—are outside this benchmark's coverage.

---

## 4. Citation Metadata Grounding vs. Semantic Entailment

- **Metadata Attribution:** Citations in this system are strictly grounded in document structural metadata extracted via PyMuPDF (`[Section Name, p. X]`). The citation validator guarantees that referenced section titles and page numbers strictly exist in the retrieved text chunks passed to the LLM.
- **Entailment Verification:** The system does not run a secondary Natural Language Inference (NLI) classifier (such as RoBERTa-large-MNLI) to formally verify premise-to-hypothesis entailment for every individual sentence clause. While grounding in retrieved context is enforced via prompt constraints and relevance thresholds, hallucination resilience relies on prompt engineering and temperature=0 settings.

---

## 5. Router Classification Boundaries

- **Router Performance:** The LangGraph router node classifies user intent into:
  - `qa` (Document Question Answering)
  - `summarize` (Executive Document Summary)
  - `notes` (Comprehensive Study Notes)
  - `outline` (Multi-Slide Presentation Outline)
  - `clarification` (Off-topic or ambiguous queries)
- **Edge-Case Ambiguity:** Queries that blend intents (e.g., *"Summarize the experimental setup in 5 presentation slides"*) can trigger edge-case router classifications. The router is biased toward `outline` when slide counts are mentioned, but multi-intent conversational compound queries may require user clarification.

---

## 6. Resource Footprint and Concurrency

- **Peak Ingestion Memory:** Processing and embedding a 15-page PDF with `all-MiniLM-L6-v2` peaks at **~923 MiB RAM**. Running on low-memory VPS environments (< 1 GB RAM) without swap will lead to Docker container termination by the Linux OOM killer.
- **Relational Concurrency:** The backend uses SQLite with WAL (Write-Ahead Logging) mode. SQLite is optimal for single-node deployments and individual researchers, but high-concurrency multi-user write bursts (> 50 simultaneous uploads) will experience file lock contention.
