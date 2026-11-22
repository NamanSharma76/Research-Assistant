# Academic Research & Learning Assistant

An agentic academic research and learning assistant grounded in scientific literature via LangGraph multi-agent orchestration, dense vector RAG (PyMuPDF, Sentence-Transformers, ChromaDB), and Groq LLMs.

---

## 1. System Architecture

```
+-----------------------------------------------------------------------------------+
|                                Streamlit Frontend                                 |
|  - Auth / Session Management (JWT Bearer Token)                                   |
|  - Document Management (Upload, Status Polling, Delete)                           |
|  - Interactive Research Chat with Real-Time LangGraph Execution Traces             |
|  - Structured Study Notes Viewer (Key Concepts, Findings, Limitations)            |
|  - Presentation Outline Viewer (Multi-Slide Decks with Bulleted Talking Points)   |
|  - Executive Summary Dashboard                                                    |
+-----------------------------------------------------------------------------------+
                                          │ HTTP / REST
                                          ▼
+-----------------------------------------------------------------------------------+
|                                 FastAPI Backend                                   |
|  - Endpoints: /auth, /documents, /chat, /documents/{id}/summary, notes, outline   |
|  - User Data Isolation (Multi-Tenant Document Scoping by user_id)                 |
|  - Idempotent SQLite Artifact Cache (Cold Synthesis vs. Warm Retrieval <15ms)     |
+-----------------------------------------------------------------------------------+
         │                                              │
         ▼                                              ▼
+──────────────────────────────+              +─────────────────────────────────────+
|   LangGraph Agent Pipeline   |              |         Deterministic RAG           |
|  - router_node (Intent)      |              |  - PyMuPDF Block & Section Parsing  |
|  - qa_retrieve & qa_answer   |              |  - Token-Aware Overlapping Chunks   |
|  - qa_refuse (Relevance)     |              |  - all-MiniLM-L6-v2 Embeddings      |
|  - notes_node, outline_node  |              |  - ChromaDB Vector Store            |
|  - summarize_node            |              |  - Strict Structural Citations:     |
|  - update_memory (History)   |              |    [Section Name, p. X]             |
+──────────────────────────────+              +─────────────────────────────────────+
```

---

## 2. Directory Structure

```
academic-assistant/
  backend/
    app/
      agents/
        graph.py           # Compiled LangGraph workflow and state schema
        nodes.py           # Individual agent nodes (router, qa, notes, outline, etc.)
        prompts.py         # Versioned agent prompts
      api/
        main.py            # FastAPI application factory and routing
        auth.py            # JWT token creation and PBKDF2 authentication
        routes/            # REST API endpoints (auth, documents, chat, etc.)
      db/
        database.py        # SQLite engine with Write-Ahead Logging (WAL)
        models.py          # SQLAlchemy ORM models (User, Document, Turn, Cache)
      rag/
        ingest.py          # PDF structural decomposition, token chunking, and embedding
        retrieval.py       # Dense vector similarity search with relevance filtering
        embeddings.py      # Sentence-Transformers all-MiniLM-L6-v2 singleton
      llm.py               # Groq LLM client wrapper with retry handling
      config.py            # Centralized application configuration
    scripts/
      export_graph.py      # Generates Mermaid diagram of LangGraph workflow
      run_backend.bat      # Windows batch runner for local FastAPI
      run_backend.sh       # Linux/macOS shell runner for local FastAPI
    tests/                 # 79 automated backend unit and integration tests
  frontend/
    app.py                 # Streamlit application entry point
    api_client.py          # HTTP client for backend REST API
    views/
      chat.py              # Chat view with trace viewer and page citations
      notes.py             # Structured notes view with defensive item rendering
      outline.py           # Multi-slide presentation outline view
      summary.py           # Document executive summary view
    scripts/
      run_frontend.bat     # Windows batch runner for local Streamlit
      run_frontend.sh      # Linux/macOS shell runner for local Streamlit
    tests/                 # 45 automated frontend unit and view tests
  docs/
    graph.mmd              # Exported Mermaid workflow definition
    graph.md               # Markdown rendering of LangGraph workflow
  SPEC.md                  # System architecture and data flow specification
  requirements.txt         # Production and development dependencies
```

---

## 3. Setup & Execution

### Option A: Run via Docker Compose (Recommended)

From the repository root:
```powershell
# Copy environment configuration
copy .env.example .env

# Start backend and frontend containers
docker compose up -d

# Check service health
docker compose ps
```
- **Frontend UI:** [http://localhost:8501](http://localhost:8501)
- **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### Option B: Run Locally Without Docker

#### Step 1: Create Virtual Environment and Install Dependencies
```powershell
cd academic-assistant
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

#### Step 2: Configure Environment Variables
Copy `.env.example` from repository root:
```powershell
copy ..\.env.example .env
```
Ensure `GROQ_API_KEY` and `JWT_SECRET` (minimum 32 characters) are set.

#### Step 3: Run Backend Service
```powershell
# In terminal 1 (from repository root or backend/scripts):
.\academic-assistant\backend\scripts\run_backend.bat
```
The FastAPI server will start on `http://127.0.0.1:8000`.

#### Step 4: Run Frontend Service
```powershell
# In terminal 2 (from repository root or frontend/scripts):
.\academic-assistant\frontend\scripts\run_frontend.bat
```
The Streamlit interface will start on `http://127.0.0.1:8501`.

---

## 4. Running Automated Tests

All tests are designed to execute offline without requiring active external network connections or live Groq API keys:

```powershell
# 1. Run Backend Tests (79 tests)
cd academic-assistant/backend
pytest tests -v

# 2. Run Frontend Tests (45 tests)
cd academic-assistant/frontend
pytest tests -v
```
