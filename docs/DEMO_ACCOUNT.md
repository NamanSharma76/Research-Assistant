# Demonstration Account Guide

This document describes the pre-configured demonstration account provided for academic evaluations, project vivas, and local architectural walkthroughs of the **Agentic AI-Based Academic Research and Learning Assistant**.

---

## 1. Demo Account Credentials & Environment Setup

To allow immediate, offline, or low-latency demonstration without requiring examiners to upload and re-index papers during the viva, a dedicated demo profile is provisioned in the persistent local database:

- **Email Address:** `demo_researcher@university.edu`
- **Environment:** Local Docker deployment (`http://localhost:8501`) or non-Docker local environment.
- **Account Role:** Standard Researcher with full document ingestion, multi-agent chat, and synthesis permissions.

> [!IMPORTANT]
> **Credential Security Notice:** In compliance with security best practices, raw production passwords and secret keys are never committed to version control. The demo account password can be set during local environment initialization via the backend registration endpoint or specified in your local `.env` configuration. For local testing, ensure your local environment uses a compliant strong password (minimum 8 characters with mixed case, numbers, and symbols).

---

## 2. Pre-Loaded Demonstration Assets

When accessing the system under this demo account, the following three benchmark research papers are indexed in the vector database with pre-cached synthesis artifacts:

1. **`attention_is_all_you_need.pdf`** (Vaswani et al., 2017)
   - Domain: Natural Language Processing / Deep Learning
   - Pre-generated: Executive Summary, Comprehensive Study Notes, 8-Slide Presentation Outline, Vector Embeddings.
2. **`leafscan_test_paper.pdf`** (LeafScan Plant Disease Diagnosis, 2024)
   - Domain: Agricultural Computer Vision / Mobile Deep Learning
   - Pre-generated: Executive Summary, Structured Notes, 8-Slide Presentation Outline, Vector Embeddings.
3. **`gridtcn_test_paper.pdf`** (GridTCN Power Load Forecasting, 2024)
   - Domain: Smart Grid Energy / Temporal Convolutional Networks
   - Pre-generated: Executive Summary, Structured Notes, 8-Slide Presentation Outline, Vector Embeddings.

---

## 3. How to Access During Demonstration

1. **Start the System:**
   ```powershell
   docker compose up -d
   ```
2. **Open the Web Interface:**
   Navigate to [http://localhost:8501](http://localhost:8501) in your browser.
3. **Log In:**
   - In the sidebar, select **Login**.
   - Enter `demo_researcher@university.edu` and your locally configured password.
   - Click **Sign In**.
4. **Instant Demonstration:**
   - Switch between papers in the sidebar dropdown.
   - Click **Summary**, **Notes**, or **Outline** to demonstrate instantaneous retrieval (< 15ms) from the SQLite artifact cache.
   - Enter questions in the chat box to demonstrate real-time LangGraph multi-agent routing, live citation generation, and agent execution traces.
