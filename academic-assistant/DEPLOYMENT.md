# Cloud Deployment Guide: Streamlit Community Cloud + Backend Container

This guide outlines how to deploy the **Academic Research & Learning Assistant** to production with:
1. **Frontend**: Hosted for free on [Streamlit Community Cloud](https://streamlit.io/cloud).
2. **Backend**: Hosted on any container platform with persistent disk (e.g. [Railway](https://railway.app), [Render](https://render.com), [Fly.io](https://fly.io), or a VPS like AWS/DigitalOcean).

---

## 🏗️ Architecture Overview

```
 [User Browser] 
       │
       ▼ (HTTPS)
 [Streamlit Community Cloud]  ──(API_URL: HTTPS REST)──▶  [Backend Container (FastAPI)]
 (Frontend: app.py)                                       ├── LangGraph Agents & Groq API
                                                          └── Persistent Volume (/app/backend/data)
                                                               ├── app.db (SQLite)
                                                               ├── chroma/ (Vector DB)
                                                               ├── uploads/ (PDF papers)
                                                               └── cache/ (Generated notes)
```

> [!IMPORTANT]
> Because Streamlit Community Cloud operates over **HTTPS**, web browsers will block insecure HTTP mixed-content calls. Your backend container **must provide an HTTPS URL** (Railway, Render, and Fly.io automatically issue free SSL certificates).

---

## 🚀 Step 1: Deploy the Backend Container

Choose one of the container hosting options below:

### Option A: Railway (Fastest & Simplest)
1. Sign up / Log in to [Railway.app](https://railway.app).
2. Click **New Project** ➔ **Deploy from GitHub repo** ➔ Select `NamanSharma76/Research-Assistant`.
3. In **Settings**:
   - **Dockerfile Path**: Set to `academic-assistant/Dockerfile.backend`.
4. In **Variables**:
   - `GROQ_API_KEY`: Your Groq API key (`gsk_...`).
   - `JWT_SECRET`: A secure random string (at least 32 characters).
   - `CORS_ORIGINS`: `https://*.streamlit.app,http://localhost:8501`.
5. In **Volumes**:
   - Click **Add Volume**, set Mount Path to `/app/backend/data`. This preserves your ChromaDB vector index and SQLite database between redeployments!
6. In **Settings** ➔ **Networking**:
   - Click **Generate Domain** (e.g. `https://academic-assistant-production.up.railway.app`).
   - Copy this URL.

---

### Option B: Render
1. Sign up / Log in to [Render.com](https://render.com).
2. Click **New +** ➔ **Web Service** ➔ Connect your GitHub repository.
3. Configure the service:
   - **Runtime**: Docker
   - **Dockerfile Path**: `./academic-assistant/Dockerfile.backend`
   - **Instance Type**: Starter (requires persistent disk).
4. Under **Disks**:
   - Add a disk mounted at `/app/backend/data` (size: 5 GB).
5. Under **Environment Variables**:
   - `GROQ_API_KEY`: Your Groq API Key.
   - `JWT_SECRET`: Random 32+ character string.
   - `CORS_ORIGINS`: `https://*.streamlit.app,http://localhost:8501`.
6. Deploy! Render will provide an HTTPS URL like `https://academic-assistant-backend.onrender.com`.

---

### Option C: Any Linux Cloud VPS (AWS EC2 / DigitalOcean / Hetzner)
If you prefer running on your own VM using Docker Compose:
```bash
git clone https://github.com/NamanSharma76/Research-Assistant.git
cd Research-Assistant
cp academic-assistant/.env.example .env
# Edit .env and fill in GROQ_API_KEY and a secure JWT_SECRET
docker compose up -d backend
```
*(Use Caddy or Nginx with Let's Encrypt to expose `https://your-domain.com` with auto-SSL).*

---

## 🎨 Step 2: Deploy Frontend on Streamlit Community Cloud

Once your backend is running and you have its HTTPS URL:

1. Sign in to [share.streamlit.io](https://share.streamlit.io/) using your GitHub account.
2. Click **New app**.
3. Fill in the app settings:
   - **Repository**: `NamanSharma76/Research-Assistant`
   - **Branch**: `phase-4-api-frontend` (or `main` after merging the PR)
   - **Main file path**: `academic-assistant/frontend/app.py`
   - **App URL**: Choose a custom subdomain (e.g. `academic-research-assistant.streamlit.app`).
4. Click **Advanced settings...**:
   - In the **Secrets** text box, add:
     ```toml
     API_URL = "https://your-backend-domain.up.railway.app"
     ```
     *(Replace with your real backend HTTPS URL).*
5. Click **Deploy!**

Streamlit Community Cloud will install dependencies from `academic-assistant/frontend/requirements.txt`, launch the application, and connect securely to your backend container!

---

## ✅ Step 3: Verification Checklist

1. Open your Streamlit Community Cloud URL (`https://your-app.streamlit.app`).
2. **Register a User**: Create a research account.
3. **Upload Paper**: Upload `attention_is_all_you_need.pdf`. The UI will poll status and transition to `🟢 Ready`.
4. **Interactive Chat**: Ask questions, inspect real citation chips, and check agent traces.
5. **Notes & Outline**: Generate notes and presentation slide decks.
