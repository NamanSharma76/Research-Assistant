"""Q&A logic answering questions grounded in retrieved paper excerpts."""
from typing import List, Dict, Any, Optional
from app.rag.prompts import QA_SYSTEM_PROMPT, format_excerpt_header, is_refusal
from app.rag.citations import verify_citations, strip_unsupported_citations
from app.llm import generate


def build_qa_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Builds user prompt containing context excerpts and student question."""
    excerpts_text = ""
    for idx, c in enumerate(chunks, 1):
        header = format_excerpt_header(idx, c["section"], c["page_start"], c["page_end"])
        excerpts_text += f"\n--- Excerpt {header} ---\n{c['text']}\n"

    return f"""Question:
{question}

Retrieved Excerpts:
{excerpts_text}

Answer with Citations:"""


def answer_question(
    doc_id: str,
    question: str,
    chunks: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Answers a question given retrieved paper chunks with strict citation verification.

    Returns:
        dict: {
            "answer": str (clean answer or refusal),
            "raw_answer": str (unmodified model response),
            "refused": bool,
            "citation_check": dict {"cited": [...], "supported": [...], "unsupported": [...]},
            "chunks": list of chunks used,
        }
    """
    if not chunks:
        refusal_msg = "The paper does not appear to cover this."
        return {
            "answer": refusal_msg,
            "raw_answer": refusal_msg,
            "refused": True,
            "citation_check": {"cited": [], "supported": [], "unsupported": []},
            "chunks": [],
        }

    user_prompt = build_qa_prompt(question, chunks)
    raw_answer = generate(prompt=user_prompt, system=QA_SYSTEM_PROMPT, model=model)

    refused = is_refusal(raw_answer)
    if refused:
        return {
            "answer": raw_answer.strip(),
            "raw_answer": raw_answer,
            "refused": True,
            "citation_check": {"cited": [], "supported": [], "unsupported": []},
            "chunks": chunks,
        }

    # Verify citations
    citation_check = verify_citations(raw_answer, chunks)
    unsupported = citation_check.get("unsupported", [])

    if unsupported:
        cleaned = strip_unsupported_citations(raw_answer, unsupported)
        final_answer = cleaned + " (unverified citations removed)"
    else:
        final_answer = raw_answer

    return {
        "answer": final_answer.strip(),
        "raw_answer": raw_answer,
        "refused": False,
        "citation_check": citation_check,
        "chunks": chunks,
    }
