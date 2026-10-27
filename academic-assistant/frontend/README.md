# Academic Research & Learning Assistant - Frontend

A Streamlit web interface for the Academic Research & Learning Assistant. The frontend communicates exclusively with the FastAPI backend over HTTP using `requests`. It does not import or execute any backend code, agents, RAG components, ChromaDB, or SQLite databases directly.

---

## 🛠️ Prerequisites & Setup

Ensure the virtual environment is created and active with the required dependencies:

```bash
# From repository root
cd academic-assistant/frontend
pip install -r requirements.txt
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:8000` | Base URL of the running FastAPI backend server |

---

## 🚀 Running the Application

### 1. Start the Backend API
First, start the FastAPI backend from `academic-assistant/backend/`:

```bash
cd academic-assistant/backend
python -m uvicorn app.api.main:app --port 8000
```

### 2. Start the Frontend
In a separate terminal, launch the Streamlit frontend:

```bash
cd academic-assistant/frontend
streamlit run app.py
```

The Streamlit UI will open in your browser at `http://localhost:8501`.

---

## 🧩 Architectural Layout

```
frontend/
  app.py                 # Page configuration, auth gate, sidebar navigation, tabs
  api_client.py          # Decoupled HTTP client talking to backend REST endpoints
  components.py          # Citation chips, trace viewer, status badges, formatters
  requirements.txt       # Streamlit and requests dependencies
  views/
    chat.py              # Interactive chat session, conversation starters, trace toggle
    notes.py             # Structured revision notes, key concepts, revision Q&A
    outline.py           # Presentation deck generator (5-15 slides) with Markdown/text downloads
    summary.py           # Executive summary and per-section breakdowns
  tests/
    test_api_client.py   # Unit tests for HTTP client, timeouts, and error handling
    test_components.py   # Unit tests for citation regex, HTML escaping, and formatters
```

---

## ⚠️ Notes & Limitations

### Browser Refresh & Session State
Streamlit stores session state (`st.session_state`) in server memory bound to the active WebSocket session. If you manually refresh or hard-reload the browser tab (`F5` or `Ctrl+R`), the WebSocket connection resets, creating a fresh session state. You will be prompted to log in again. This is standard Streamlit architecture.
