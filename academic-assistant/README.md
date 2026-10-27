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
Run the complete test suite across all phases from `backend`:
```bash
cd backend
..\venv\Scripts\pytest tests -v
```

---

## 5. Phase 4: HTTP API & Web Server

### Running the API Server
Start the Uvicorn ASGI server with reload:
```bash
cd backend
..\venv\Scripts\uvicorn app.api.main:app --reload --port 8000
```

- **Interactive API Documentation (Swagger)**: Visit `http://localhost:8000/docs`
- **Alternative Documentation (ReDoc)**: Visit `http://localhost:8000/redoc`
- **Health Check**: `GET http://localhost:8000/health`

---

## 6. Phase 4.5: Streamlit Web UI

A full interactive frontend built in Streamlit communicating exclusively with the FastAPI backend over HTTP.

### Running the Frontend
```bash
cd frontend
..\venv\Scripts\streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Key Features
- **User Authentication**: Register and log in with JWT tokens.
- **Document Management**: Upload scientific PDFs, monitor background ingestion progress in real-time, and manage papers.
- **Multi-Turn Chat**: Ask questions grounded in paper excerpts with interactive citation chips and collapsible agent execution traces.
- **Study Notes**: Review generated study notes with key terms, definitions, and revision questions.
- **Slide Deck Outlines**: Configure presentation slide count (5–15 slides) and download presentation decks in Markdown or plain text.
- **Executive Summaries**: High-level synthesis alongside per-section summaries.

### Running Frontend Tests
```bash
..\venv\Scripts\pytest frontend/tests -v
```


