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
