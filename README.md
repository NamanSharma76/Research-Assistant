# Agentic AI-Based Academic Research and Learning Assistant

An agentic academic research and learning assistant built with FastAPI, Streamlit, LangGraph, ChromaDB, and Groq LLMs. The system provides grounded research synthesis, cited question answering, structured note-taking, presentation outline generation, and executive summaries over academic papers.

---

## Run with Docker

The application is containerized with Docker and Docker Compose for reproducible, zero-configuration local and production deployment.

railway.json and render.yaml are untested; the supported deployment is Docker Compose (see the free-tier memory limits in the troubleshooting section).

### Prerequisites

- **Docker Desktop** installed with the **WSL 2 backend** (Windows) or Docker Engine + Docker Compose v2 (Linux / macOS).
- Docker Engine must be running before executing compose commands.

### Quick Start

1. **Clone the repository:**
   ```powershell
   git clone <repository-url>
   cd Research-Assistant
   ```

2. **Configure environment variables:**
   Copy `.env.example` to `.env` in the repository root:
   ```powershell
   # PowerShell (Windows)
   copy .env.example .env

   # Bash (Linux / macOS)
   cp .env.example .env
   ```

3. **Set required secrets in `.env`:**
   Open `.env` and set:
   - `GROQ_API_KEY`: Your Groq API key from [https://console.groq.com/keys](https://console.groq.com/keys).
   - `JWT_SECRET`: A secure random string (minimum 32 characters), e.g. generated via:
     ```bash
     python -c "import secrets; print(secrets.token_hex(32))"
     ```

4. **Build and start services:**
   ```powershell
   docker compose up --build -d
   ```

5. **Access the application:**
   - **Streamlit Web UI:** [http://localhost:8501](http://localhost:8501)
   - **FastAPI Interactive Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Backend Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

### Docker Compose Commands

| Action | Command | Description |
| :--- | :--- | :--- |
| **Start Services** | `docker compose up -d` | Starts backend and frontend in the background |
| **Rebuild & Start** | `docker compose up --build -d` | Rebuilds container images and starts services |
| **Check Status** | `docker compose ps` | Displays service status and health checks |
| **View All Logs** | `docker compose logs -f` | Streams real-time logs from both containers |
| **View Backend Logs** | `docker compose logs -f backend` | Streams backend logs (Uvicorn / FastAPI) |
| **View Frontend Logs** | `docker compose logs -f frontend` | Streams frontend logs (Streamlit) |
| **Stop Services** | `docker compose down` | Stops containers while preserving database & data |
| **Reset Data** | `docker compose down -v` | Stops containers and deletes the `assistant_data` volume |

---

### Data Persistence

All persistent state is stored in a single named volume: **`assistant_data`**, mounted to `/app/data` inside the backend container:
- **SQLite Database:** `/app/data/app.db` (user accounts, document metadata, chat turns)
- **Vector Database:** `/app/data/chroma/` (ChromaDB embeddings & chunk indices)
- **PDF Uploads:** `/app/data/uploads/` (uploaded research papers organized by user ID)
- **Artifact Cache:** `/app/data/cache/` (cached notes, presentation outlines, and summaries)

Stopping containers with `docker compose down` retains all documents, user accounts, and caches. To start from a completely clean slate, run `docker compose down -v`.

---

### Troubleshooting

#### 1. Port already in use (`8000` or `8501`)
- **Error:** `Bind for 0.0.0.0:8000 failed: port is already allocated`
- **Solution:** Another process is using port 8000 or 8501 (such as a local non-Docker Uvicorn/Streamlit instance).
  - On Windows PowerShell, find and terminate the occupying process:
    ```powershell
    Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
    Get-Process -Id (Get-NetTCPConnection -LocalPort 8501).OwningProcess | Stop-Process -Force
    ```

#### 2. Missing Environment Variables (`GROQ_API_KEY` or `JWT_SECRET`)
- **Error:** Backend fails on startup with `RuntimeError: GROQ_API_KEY environment variable is missing` or `JWT_SECRET environment variable is missing or shorter than 32 characters`.
- **Solution:** Ensure `.env` exists in the repository root and contains valid `GROQ_API_KEY` and `JWT_SECRET` (32+ characters). Restart with `docker compose up -d`.

#### 3. Docker Out of Memory (OOM)
- **Symptoms:** Container exits with code 137 during embedding or PyTorch operations.
- **Resource Requirement:** The measured peak backend memory during document ingestion and multi-agent synthesis is **923 MiB**. Free-tier cloud instances with 512 MiB will experience OOM kills.
- **Solution:** Increase Docker Desktop resource limits:
  1. Open **Docker Desktop Settings** -> **Resources** -> **Advanced**.
  2. Increase Memory allocation to at least 4 GB.
  3. Click **Apply & Restart**.

#### 4. Sentence-Transformers Offline / Cold Start
- The `all-MiniLM-L6-v2` embedding model is pre-downloaded during `docker compose build` into `/app/.cache/huggingface`. The backend requires no internet access to load embedding weights at runtime.
