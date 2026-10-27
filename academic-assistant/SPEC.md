# Architecture Specification: Academic Research and Learning Assistant

## 1. Project Goal
The **Agentic AI Based Academic Research and Learning Assistant** is a multi-agent system (built with LangGraph in future phases) where students upload academic research paper PDFs and specialized agents assist in comprehension and study workflows. The agents answer questions, summarize content, generate structured notes, and draft presentation outlines—all strictly grounded in the paper content via Retrieval-Augmented Generation (RAG) backed by ChromaDB and Groq LLMs.

This specification documents **Phase 1: Project Foundation & Core RAG Pipeline**.

---

## 2. Overall Multi-Agent Architecture (Future Phases & Foundation)

The system is designed around a shared RAG and document layer, topped by specialized autonomous agents orchestrated via LangGraph:

```
                  +-----------------------------------+
                  |         Student / Client          |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |           Router Agent            |
                  +-----------------+-----------------+
                                    |
         +-----------------+--------+--------+-----------------+
         |                 |                 |                 |
         v                 v                 v                 v
+-----------------+ +---------------+ +---------------+ +---------------+
|  Retrieval/QA   | |  Summarizer   | | Notes Agent   | | Outline Agent |
|     Agent       | |     Agent     | |               | |               |
+--------+--------+ +-------+-------+ +-------+-------+ +-------+-------+
         |                  |                 |                 |
         +------------------+--------+--------+-----------------+
                                     |
                                     v
                  +-----------------------------------+
                  |      Shared RAG Engine Layer      |
                  |  - Ingestion (PyMuPDF)            |
                  |  - Embeddings (all-MiniLM-L6-v2)  |
                  |  - Vector Store (ChromaDB)        |
                  |  - Retrieval APIs                 |
                  +-----------------------------------+
```

### Agents Overview
1. **Router Agent**: Analyzes user intent and dispatches queries to the appropriate specialized downstream agent.
2. **Retrieval / QA Agent**: Answers specific factual and conceptual questions using semantic search over chunk embeddings.
3. **Summarizer Agent**: Generates executive summaries, section-by-section breakdowns, or high-level overviews using retrieved section chunks.
4. **Notes Agent**: Synthesizes key takeaways, formulas, definitions, and learning notes formatted for study.
5. **Outline Agent**: Produces slide decks or presentation outlines organized logically based on the paper's narrative and sections.

---

## 3. Core Retrieval Functions

All downstream agents interface with the vector store through three canonical functions defined in `backend/app/rag/retrieval.py`:

### `search(doc_id: str, query: str, k: int = 5) -> list[dict]`
- **Purpose**: Semantic search over chunks belonging strictly to the specified `doc_id`.
- **Mechanism**: Queries ChromaDB with similarity search restricted by metadata filter `{"doc_id": doc_id}`.
- **Output**: List of dictionaries containing:
  - `text` (str): Chunk textual content.
  - `section` (str): Section name the chunk belongs to.
  - `page_start` (int): 1-indexed starting page number.
  - `page_end` (int): 1-indexed ending page number.
  - `score` (float): Normalized similarity score between `0.0` and `1.0` (where 1.0 represents highest similarity).

### `get_section(doc_id: str, section_name: str) -> list[dict]`
- **Purpose**: Direct metadata-filtered retrieval of all chunks within a specific section (e.g., "Abstract", "Method", "Results").
- **Mechanism**: ChromaDB metadata filtering with `{"$and": [{"doc_id": doc_id}, {"section": section_name}]}`, ordered ascending by `chunk_index`. Does not perform similarity scoring.
- **Output**: List of chunk dictionaries sorted in original document order.

### `get_all_chunks(doc_id: str) -> list[dict]`
- **Purpose**: Retrieve the entire document chunk stream.
- **Mechanism**: ChromaDB query filtered by `{"doc_id": doc_id}`, ordered ascending by `chunk_index`.
- **Output**: Complete list of chunks representing the whole document in chronological reading sequence.

---

## 4. Chunk Metadata Schema

Every chunk ingested into ChromaDB contains the following strict metadata attributes:

| Field | Type | Description |
|---|---|---|
| `doc_id` | `str` | Short SHA-256 hash (first 12 characters) of the file content. Ensures idempotency. |
| `filename` | `str` | Name of the source PDF file (e.g., `attention_is_all_you_need.pdf`). |
| `section` | `str` | Detected section header (e.g., `Abstract`, `Introduction`, `Method`). Defaults to `"Unknown"`. |
| `page_start` | `int` | First PDF page (1-indexed) where this chunk begins. |
| `page_end` | `int` | Last PDF page (1-indexed) where this chunk ends. |
| `chunk_index` | `int` | Sequential chunk index within the document (`0, 1, 2, ...`). |
| `chunk_id` | `str` | Unique chunk identifier formatted as `{doc_id}_c{chunk_index:04d}`. |

---

## 5. Strict Grounding and Citation Rules

To eliminate hallucinations and maintain scientific integrity:
1. **Retrieved-Only Grounding**: Agent answers must strictly and exclusively use information present in the retrieved chunk text. External pre-trained knowledge must never substitute or contradict the paper text.
2. **Explicit Citations**: Every factual statement, claim, or synthesized point must cite the section name and page range in the format:
   `[Section, p. X]` or `[Section, pp. X-Y]`.
   *Example*: `The Transformer architecture relies entirely on self-attention mechanisms without recurrence [Introduction, p. 2].`
3. **Relevance Thresholding**: If the top similarity score is below the configured threshold (e.g., `0.35`), the system must reject answering and report:
   `"The paper does not appear to cover this."` without making an LLM API call.

---

## 6. Phase 4 HTTP API & Database Architecture

Phase 4 wraps the core RAG and LangGraph agent pipelines into a production-grade FastAPI HTTP service with SQLite persistence, user isolation, and JWT authentication.

### Database Schema (SQLite + SQLAlchemy ORM)
- **User**: `id` (UUID), `email` (unique, lowercase), `password_hash` (bcrypt), `created_at`.
- **Document**: `id` (UUID), `user_id` (FK User, cascade), `doc_id` (Chroma content hash), `filename`, `file_path`, `status` ("processing" | "ready" | "failed"), `error`, `num_chunks`, `created_at`. Unique constraint on `(user_id, doc_id)`.
- **Conversation**: `id` (UUID), `user_id` (FK User, cascade), `document_id` (FK Document, cascade), `title`, `created_at`.
- **Message**: `id` (UUID), `conversation_id` (FK Conversation, cascade), `role` ("user" | "assistant"), `content`, `intent`, `trace` (JSON), `llm_calls`, `seconds`, `kind`, `created_at`.

### Ingestion Serialization & Background Processing
- Uploaded PDFs are streamed into `data/uploads/<user_id>/<uuid>.pdf`.
- Magic byte validation checks `%PDF` header; max size enforced at `MAX_UPLOAD_MB`.
- Ingestion runs as a FastAPI `BackgroundTask` serialized globally with a mutex lock to prevent concurrent ChromaDB/model race conditions.
- Uses its own dedicated `SessionLocal()` to decouple from HTTP connection lifecycles.

### Artifact Endpoints & Synchronous Threadpool Execution
- `GET /documents/{id}/summary?force=false`
- `GET /documents/{id}/notes?force=false`
- `GET /documents/{id}/outline?slides=10&force=false`
These calls can take tens of seconds to complete on initial generation. They are implemented as synchronous `def` endpoints so FastAPI automatically schedules them across its background threadpool without blocking event loop concurrency.

---

## 7. Phase 4.5 Streamlit Frontend Architecture

The frontend is a dedicated Streamlit application decoupled from backend internals, communicating exclusively over HTTP via standard REST endpoints using `requests`.

### Decoupled HTTP Architecture
- No direct access to agents, vector databases, ChromaDB, or SQLite models.
- Interacts strictly via `frontend/api_client.py` using `requests`.
- Handles connection errors, request timeouts, and backend 401/404/409/429/502/503 errors with user-friendly alerts.

### Interactive Components & Features
- **Authentication Gate**: Login and Register tabs with UTF-8 password byte validation (8-72 bytes) and auto-login on successful registration.
- **Sidebar Navigation**:
  - User profile with clean session logout.
  - PDF document uploader with background status polling via `@st.fragment(run_every=3)`.
  - Document selector with emoji status indicators (`🟢 Ready`, `⏳ Processing`, `🔴 Failed`).
  - Conversation manager per active document ("➕ New Chat", select, and delete).
  - Agent trace toggle to inspect LangGraph execution steps, LLM calls, and latency.
- **Interactive Workspaces (Tabs)**:
  - **💬 Chat**: Real-time conversation with optimistic user bubble rendering, starter prompts, and collapsible agent trace inspections.
  - **📖 Notes**: Structured study notes with section expanders, key concepts, revision Q&A, and interactive citation chips.
  - **📊 Outline**: Slide presentation deck generator with slide count slider (5-15 slides) and one-click Markdown / plain text downloads.
  - **📄 Summary**: Executive summary and per-section breakdowns with grounded citation chips.
- **Citation Chips & Security**:
  - Regex-based academic citation parser `[Section, p. X]` converting text citations into styled HTML chips.
  - Strict HTML escaping (`html.escape`) prior to chip insertion to prevent any XSS vulnerability.


