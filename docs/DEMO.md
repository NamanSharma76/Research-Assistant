# Final Project Demonstration & Viva Script (6-Minute Walkthrough)

This guide provides an exact, minute-by-minute demonstration script for project examiners, academic evaluators, and viva presentation panels for the **Agentic AI-Based Academic Research and Learning Assistant**.

---

## Demonstration Timeline Overview

| Timestamp | Phase | Focus Area | Key Visual / Action |
| :--- | :--- | :--- | :--- |
| **0:00 – 1:00** | Introduction & Problem Statement | Core Motivation, Dual Architecture | Architecture diagram, Streamlit dashboard |
| **1:00 – 2:30** | Interactive Agentic Q&A | Grounded RAG, Page Citations, Traces | Live query on *Attention Is All You Need* |
| **2:30 – 3:45** | Academic Synthesis Artifacts | Structured Notes & 8-Slide Outlines | Instant cache retrieval on *LeafScan* & *GridTCN* |
| **3:45 – 5:00** | Empirical Benchmark Evaluation | Baseline vs. Agentic RAG Trade-offs | Benchmark findings table, 30-question eval data |
| **5:00 – 6:00** | DevOps & Security Verification | Docker, Isolation, Persistence, Q&A | Container status, JWT auth, WAL database |

---

## Detailed Step-by-Step Viva Script

### 0:00 – 1:00 | Introduction, Motivation & System Architecture

**Presenter Script:**
> *"Good morning/afternoon, esteemed committee members. Today I am presenting our B.Tech Capstone Project: the **Agentic AI-Based Academic Research and Learning Assistant**.
>
> Modern researchers and students face severe information overload when analyzing dense scientific literature. While commercial LLMs can summarize text, they frequently suffer from ungrounded hallucinations, lack verifiable page-level citations, and lack structured workflows for academic comprehension.
>
> Our system addresses this through an **Agentic LangGraph Orchestration pipeline** paired with a deterministic Vector RAG backend. We utilize FastAPI for the backend service, Streamlit for an interactive multi-view UI, PyMuPDF for structural document decomposition, Sentence-Transformers (`all-MiniLM-L6-v2`) for local dense vector search in ChromaDB, and Groq-hosted LLaMA 3.3 70B for high-speed inference.
>
> All user state, document chunks, and synthesis artifacts are persisted in SQLite and ChromaDB, fully containerized via Docker Compose."*

**Visual Action:**
- Show the Streamlit Web UI at [http://localhost:8501](http://localhost:8501).
- Highlight the sidebar showing authenticated session as `demo_researcher@university.edu` and the pre-loaded research papers.

---

### 1:00 – 2:30 | Live Cited Question Answering & Execution Trace

**Presenter Script:**
> *"Let us demonstrate our real-time cited Q&A capabilities. I will select the seminal paper: **Attention Is All You Need** (Vaswani et al., 2017).
>
> I will ask a direct factual question regarding the training dataset used for machine translation."*

**Action:**
1. Select `attention_is_all_you_need.pdf` from the document dropdown.
2. In the chat input, type:
   ```text
   What dataset did they use for translation experiments?
   ```
3. Submit the query.

**Expected Response:**
> *"The translation experiments were conducted on the **WMT 2014 English-German dataset** consisting of about 4.5 million sentence pairs, and the **WMT 2014 English-French dataset** consisting of 36 million sentence pairs [5 Training, pp. 7-8]."*

**Presenter Script:**
> *"Notice two key architectural strengths here:
> 1. **Verifiable Page-Level Citation:** The citation `[5 Training, pp. 7-8]` directly references the exact structural section and physical PDF page.
> 2. **Execution Trace Inspection:** Expanding the agent trace shows the LangGraph execution flow: `router_node` classified the intent as `qa`, routed to `qa_retrieve` to query ChromaDB, synthesized the answer in `qa_answer`, and finally updated conversation history in `update_memory`."*

**Off-Topic / Out-of-Domain Demonstration (Autonomous Refusal):**
- Ask: *"What is the recipe for chocolate cake?"*
- **Response:** The system refuses politely: *"The uploaded document does not contain information to answer this question."* (Triggering the 0.50 cosine similarity threshold without hallucinating).

---

### 2:30 – 3:45 | Academic Synthesis: Structured Notes & 8-Slide Outline

**Presenter Script:**
> *"Beyond simple Q&A, academic researchers require multi-perspective synthesis. Our assistant provides automatic generation of structured study notes and conference presentation slide outlines."*

**Action:**
1. Switch to `leafscan_test_paper.pdf` or `gridtcn_test_paper.pdf`.
2. Click on the **Notes** tab.
   - Point out the structured sections: *Problem Statement*, *Methodology*, *Key Findings* (e.g., highlighting MAPE 4.1% or mobile model accuracy), and *Limitations*.
3. Click on the **Outline** tab.
   - Showcase the generated 8-slide presentation deck.
   - Show that Slide 1 accurately captures the paper title and domain without author hallucination.
   - Show that Slides 2 through 8 cover Problem, Architecture, Training, Results, and Future Work with bulleted talking points.

**Rate Limit Fallback Strategy:**
> *"Because artifact generation requires multiple LLM synthesis calls, we engineered an **idempotent SQLite artifact cache**. Cold generation takes ~20 seconds; however, subsequent loads are served from cache in under 15 milliseconds, ensuring completely resilient live presentations even under cloud rate limits."*

---

### 3:45 – 5:00 | Empirical Benchmark & Evaluation Findings

**Presenter Script:**
> *"A key requirement of our project was rigorous, reproducible evaluation. In Phase 7b, we developed an automated evaluation harness comparing **Baseline Single-Turn RAG** against our **Agentic LangGraph Pipeline** across 30 questions on 3 papers."*

**Action:**
- Display the comparative evaluation summary table (from `KNOWN_LIMITATIONS.md` or presentation slide):

| Metric | Baseline RAG (Single-Turn) | Agentic LangGraph RAG | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **Overall Strict Correctness** | **83.3%** (25 / 30) | **83.3%** (25 / 30) | 0.0% |
| **Partial Factual Accuracy (Mean Score)** | **90.6%** (27.17 / 30.00) | **90.6%** (27.17 / 30.00) | 0.0% |
| **Answerable Strict Accuracy (Covered)** | **76.2%** (16 / 21) | **76.2%** (16 / 21) | 0.0% |
| **Refusal Recall (True Refusals)** | **100.0%** (9 / 9) | **100.0%** (9 / 9) | 0.0% |
| **Refusal Precision** | **90.0%** (9 / 10) | **90.0%** (9 / 10) | 0.0% |
| **Citation Reference Validity** | **100.0%** (35 / 35 supported) | **96.4%** (27 / 28 supported) | -3.6% |
| **Average End-to-End Latency** | **14.03s** (P50: 11.07s) | **22.37s** (P50: 18.96s) | +8.34s (1.59x) |
| **Average LLM Calls per Turn** | **1.0 call** (30 total) | **2.6 calls** (78 total) | +1.6 calls (2.60x) |

**Presenter Script:**
> *"Our findings provide an honest academic insight: for direct, single-hop factual retrieval, baseline RAG achieves identical partial accuracy with lower latency. However, the agentic architecture provides the dynamic routing, conversational state, slide structuring, and multi-step decomposition necessary for an end-to-end learning environment."*

---

### 5:00 – 6:00 | DevOps, Security Boundaries & Panel Q&A

**Presenter Script:**
> *"Finally, we highlight production readiness and security hardening:
> 1. **Containerization:** The entire stack is orchestrated via Docker Compose with dedicated backend and frontend services, health checks, and a persistent data volume (`assistant_data`).
> 2. **Security & Authentication:** User accounts are secured using PBKDF2 password hashing and stateless JWT bearer authentication. Documents are scoped strictly by user ID to prevent unauthorized cross-tenant data access.
> 3. **Extensive Test Coverage:** We maintain 148 automated automated tests across the backend (79 tests), frontend (45 tests), and evaluation harness (24 tests), with 100% pass rate.
>
> Thank you, and we welcome questions from the committee."*
