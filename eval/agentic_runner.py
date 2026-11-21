"""Agentic evaluation runner executing fresh isolated LangGraph turns per question."""
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

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

from app.agents.graph import compiled_app, AgentState
from app.llm import get_call_count
from app.rag.prompts import is_refusal
from app.rag.citations import parse_citations


def run_agentic_turn(
    doc_id: str,
    question: str,
) -> Dict[str, Any]:
    """Executes a single evaluation question in a fresh, isolated Agent turn.

    Captures both before-verification raw answers and after-verification clean answers,
    distinguishing pre-verification citation attempts from post-verification grounded citations.

    Args:
        doc_id: 12-char document identifier.
        question: Evaluation question text.

    Returns:
        Dict containing answer, raw_answer, intent, standalone_query, trace,
        llm_calls, seconds, refused, citation_check, pre_citations, post_citations.
    """
    initial_calls = get_call_count()
    t0 = time.time()

    initial_state: AgentState = {
        "doc_id": doc_id,
        "user_message": question,
        "history": [],
        "last_output": {"kind": "none", "text": ""},
        "decision": None,
        "chunks": [],
        "answer": "",
        "qa_result": None,
        "retries": 0,
        "trace": [],
        "result_text": "",
    }

    final_state = compiled_app.invoke(initial_state)

    elapsed = time.time() - t0
    turn_calls = get_call_count() - initial_calls

    decision = final_state.get("decision")
    intent = decision.intent if decision else "paper_question"
    standalone_q = decision.standalone_query if decision else question
    trace = final_state.get("trace", [])

    clean_answer = final_state.get("result_text", "")
    qa_res = final_state.get("qa_result") or {}

    raw_answer = qa_res.get("raw_answer", clean_answer)
    citation_check = qa_res.get("citation_check", {"cited": [], "supported": [], "unsupported": []})

    # Distinguish pre-verification citations vs post-verification citations
    pre_citations = parse_citations(raw_answer)
    post_citations = parse_citations(clean_answer)

    refused = is_refusal(clean_answer)

    return {
        "answer": clean_answer,
        "raw_answer": raw_answer,
        "intent": intent,
        "standalone_query": standalone_q,
        "trace": trace,
        "llm_calls": turn_calls,
        "seconds": elapsed,
        "refused": refused,
        "citation_check": citation_check,
        "pre_citations": pre_citations,
        "post_citations": post_citations,
    }
