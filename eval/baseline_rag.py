"""Traditional single-call top-5 baseline RAG implementation.

Baseline RAG Specification:
- Same embedding model (all-MiniLM-L6-v2) and ChromaDB vector store.
- Raw user question (no query rewriting or optimization).
- Top-5 retrieved chunks from ChromaDB.
- NO relevance threshold filtering (no pre-LLM threshold refusal).
- Exactly one main LLM call (GROQ_MODEL_MAIN) per question.
- No intent router.
- No query rewriting.
- No semantic retries.
- No citation stripping or post-processing (preserves raw model response).
- Relies strictly on QA_SYSTEM_PROMPT grounding prompt to refuse unsupported questions.
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

eval_dir = Path(__file__).resolve().parent
eval_data_dir = (eval_dir / "data").resolve()
eval_data_dir.mkdir(parents=True, exist_ok=True)
if "DATA_DIR" not in os.environ:
    os.environ["DATA_DIR"] = str(eval_data_dir)
if "CHROMA_PERSIST_DIRECTORY" not in os.environ:
    os.environ["CHROMA_PERSIST_DIRECTORY"] = str((eval_data_dir / "chroma").resolve())

backend_dir = eval_dir.parent / "academic-assistant" / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import GROQ_MODEL_MAIN, LLM_CALL_DELAY_SECONDS
from app.rag.retrieval import search
from app.rag.prompts import QA_SYSTEM_PROMPT, is_refusal
from app.rag.qa import build_qa_prompt
from app.rag.citations import verify_citations
from app.llm import generate, get_call_count


def run_baseline_rag(
    doc_id: str,
    question: str,
    k: int = 5,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes single-call top-5 baseline RAG for a question against a document.

    Args:
        doc_id: 12-char document ID hash.
        question: Raw user query string.
        k: Maximum retrieved chunks (default: 5).
        model: Optional model override (default: GROQ_MODEL_MAIN).

    Returns:
        Dict: {
            "answer": str,
            "raw_answer": str,
            "refused": bool,
            "llm_calls": int,
            "seconds": float,
            "retrieved_count": int,
            "chunks": List[Dict],
            "citation_check": Dict,
        }
    """
    target_model = model or GROQ_MODEL_MAIN
    initial_calls = get_call_count()
    t0 = time.time()

    # Step 1: Raw top-k semantic search (No threshold filtering!)
    retrieved_chunks = search(doc_id=doc_id, query=question, k=k, exclude_sections=("References",))

    # Step 2: Build QA prompt with raw retrieved chunks
    user_prompt = build_qa_prompt(question, retrieved_chunks)

    # Step 3: Exactly one generation call to main LLM
    raw_answer = generate(
        prompt=user_prompt,
        system=QA_SYSTEM_PROMPT,
        model=target_model,
        temperature=0.2,
    )
    time.sleep(LLM_CALL_DELAY_SECONDS)

    elapsed = time.time() - t0
    measured_calls = get_call_count() - initial_calls
    turn_calls = measured_calls if measured_calls > 0 else 1

    refused = is_refusal(raw_answer)

    # In baseline, citations are verified for evaluation metrics, but NOT stripped from the answer
    citation_check = verify_citations(raw_answer, retrieved_chunks) if not refused else {
        "cited": [],
        "supported": [],
        "unsupported": [],
    }

    return {
        "answer": raw_answer.strip(),
        "raw_answer": raw_answer,
        "refused": refused,
        "llm_calls": turn_calls,
        "seconds": elapsed,
        "retrieved_count": len(retrieved_chunks),
        "chunks": retrieved_chunks,
        "citation_check": citation_check,
    }
