# Academic Research & Learning Assistant (Phase 1)

Multi-agent academic research assistant foundation grounded in research papers via RAG (PyMuPDF, sentence-transformers, ChromaDB, and Groq API).

---

## 1. Project Structure

```
academic-assistant/
  backend/
    app/
      rag/
        __init__.py
        ingest.py        # PDF -> blocks & sections -> token chunks -> embeddings -> ChromaDB
        retrieval.py     # search(), get_section(), get_all_chunks()
        embeddings.py    # Singleton wrapper for sentence-transformers all-MiniLM-L6-v2
      llm.py             # Thin wrapper around Groq API
      config.py          # Centralized configuration (paths, chunk sizes, models, thresholds)
    scripts/
      ingest_pdf.py      # CLI: Ingests PDF paper into ChromaDB idempotently
      ask.py             # CLI: Cited RAG Q&A with relevance threshold filtering
    tests/
      test_rag.py        # Pytest test suite (metadata, idempotency, ordering, isolation)
    data/
      uploads/           # Research paper PDFs storage
      chroma/            # ChromaDB persistent vector database
    pytest.ini
  SPEC.md                # System specification & architectural conventions
  requirements.txt       # Dependencies
  .env.example           # Environment template
  .gitignore             # Ignores .env, venv, and data/chroma
  README.md
```

---

## 2. Setup & Installation

### Step 1: Create Virtual Environment
```bash
cd academic-assistant
python -m venv venv
.\venv\Scripts\activate   # On Windows
# source venv/bin/activate # On Linux/macOS
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Edit `.env` and add your `GROQ_API_KEY`:
```ini
GROQ_API_KEY=your_actual_groq_api_key
```

---

## 3. How to Run

### Ingest a Research Paper PDF
```bash
python backend/scripts/ingest_pdf.py backend/data/uploads/attention_is_all_you_need.pdf
```
*Note: Ingesting the same file again will check the SHA-256 `doc_id` and skip duplicates idempotently.*

### Ask Questions Over a Document
Run Q&A using the 12-character `doc_id`:
```bash
python backend/scripts/ask.py <doc_id> "<question>"
```

Examples:
```bash
# 1. Answerable Question
python backend/scripts/ask.py bdfaa68d8984 "What is the primary difference between the Transformer and previous recurrent models?"

# 2. Section-Specific Question
python backend/scripts/ask.py bdfaa68d8984 "What dataset and BLEU score were obtained in the English-to-German translation experiments?"

# 3. Question Not Covered (Relevance Threshold Trigger)
python backend/scripts/ask.py bdfaa68d8984 "What is photosynthesis in plant cells?"
```

---

## 4. Running Tests
Run the test suite from `backend`:
```bash
cd backend
..\venv\Scripts\pytest tests\test_rag.py -v
```
Tests cover:
- Chunk metadata retention (`doc_id`, `filename`, `section`, `page_start`, `page_end`, `chunk_index`, `chunk_id`).
- Idempotent re-ingestion without chunk duplication.
- `get_section` filtered and strictly ordered by `chunk_index`.
- `search` isolation respecting `doc_id` boundaries.
