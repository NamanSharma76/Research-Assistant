"""Section and whole-document summarization agents."""
import time
from typing import Dict, Any, List
from app.config import (
    GROQ_MODEL_MAIN,
    GROQ_MODEL_FAST,
    MAX_PROMPT_TOKENS,
    LLM_CALL_DELAY_SECONDS,
)
from app.llm import generate, estimate_tokens
from app.rag.retrieval import get_section, get_all_chunks
from app.rag.citations import verify_citations, strip_unsupported_citations
from app.agents import cache
from app.agents.prompts import (
    SECTION_SUMMARY_PROMPT,
    MAP_SUMMARY_PROMPT,
    REDUCE_SUMMARY_PROMPT,
    DOCUMENT_OVERALL_SUMMARY_PROMPT,
)
from app.agents.cleaning import strip_meta_from_summary

NON_CONTENT_SECTIONS = {
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "unknown",
}


def _format_chunk_with_page(chunk: dict) -> str:
    p_start = chunk.get("page_start", 0)
    p_end = chunk.get("page_end", 0)
    p_str = f"p. {p_start}" if p_start == p_end else f"pp. {p_start}-{p_end}"
    return f"[{chunk.get('section', 'Unknown')}, {p_str}]\n{chunk.get('text', '')}"


def summarize_section(doc_id: str, section_name: str) -> str:
    """Summarizes a specific document section.
    Uses direct single-prompt if token count <= MAX_PROMPT_TOKENS,
    otherwise performs map-reduce chunk summarization.
    """
    chunks = get_section(doc_id, section_name)
    if not chunks:
        return f"Section '{section_name}' not found or has no content."

    # Format all section chunks with citation headers
    formatted_chunks = [_format_chunk_with_page(c) for c in chunks]
    combined_text = "\n\n".join(formatted_chunks)

    # Check token count
    token_est = estimate_tokens(combined_text)

    if token_est <= MAX_PROMPT_TOKENS:
        # Single-prompt summarization
        prompt = SECTION_SUMMARY_PROMPT.format(section_name=section_name, text=combined_text)
        raw_summary = generate(prompt, model=GROQ_MODEL_FAST)
        time.sleep(LLM_CALL_DELAY_SECONDS)
    else:
        # Map-Reduce summarization
        # Map phase: summarize each chunk individually
        map_summaries = []
        for fc in formatted_chunks:
            map_prompt = MAP_SUMMARY_PROMPT.format(section_name=section_name, text=fc)
            part = generate(map_prompt, model=GROQ_MODEL_FAST)
            time.sleep(LLM_CALL_DELAY_SECONDS)
            map_summaries.append(part.strip())

        # Reduce phase: combine partial summaries
        reduce_input = "\n\n".join(map_summaries)
        reduce_prompt = REDUCE_SUMMARY_PROMPT.format(section_name=section_name, text=reduce_input)
        raw_summary = generate(reduce_prompt, model=GROQ_MODEL_FAST)
        time.sleep(LLM_CALL_DELAY_SECONDS)

    # Verify and clean citations against the actual chunks of this section
    verified = verify_citations(raw_summary, chunks)
    if verified["unsupported"]:
        clean_summary = strip_unsupported_citations(raw_summary, verified["unsupported"])
    else:
        clean_summary = raw_summary

    return clean_summary.strip()


def summarize_document(doc_id: str, force: bool = False) -> Dict[str, Any]:
    """Generates an executive summary and per-section breakdowns for a paper.
    Caches result to data/cache/<doc_id>/summary.json.
    """
    if not force:
        cached = cache.get(doc_id, "summary")
        if cached is not None:
            return cached

    all_chunks = get_all_chunks(doc_id)
    if not all_chunks:
        return {"overall": "No document chunks found.", "sections": {}}

    # Preserve section appearance order and group chunks
    sections_order: List[str] = []
    seen = set()
    for c in all_chunks:
        sec = c.get("section", "Unknown")
        if sec not in seen:
            seen.add(sec)
            sections_order.append(sec)

    # Filter out non-content sections
    content_sections = [
        sec for sec in sections_order
        if sec.strip().lower() not in NON_CONTENT_SECTIONS
    ]

    section_summaries: Dict[str, str] = {}
    for sec in content_sections:
        sec_summary = summarize_section(doc_id, sec)
        section_summaries[sec] = sec_summary

    # Overall synthesis using GROQ_MODEL_MAIN
    joined_summaries = "\n\n".join(
        f"### {sec}\n{summ}" for sec, summ in section_summaries.items()
    )
    overall_prompt = DOCUMENT_OVERALL_SUMMARY_PROMPT.format(section_summaries=joined_summaries)
    raw_overall_summary = generate(overall_prompt, model=GROQ_MODEL_MAIN)
    time.sleep(LLM_CALL_DELAY_SECONDS)

    # Collect union of all chunks belonging to content sections for citation verification
    content_chunks = [
        c for c in all_chunks
        if c.get("section", "Unknown").strip().lower() not in NON_CONTENT_SECTIONS
    ]

    verified = verify_citations(raw_overall_summary, content_chunks)
    if verified["unsupported"]:
        clean_overall = strip_unsupported_citations(raw_overall_summary, verified["unsupported"])
    else:
        clean_overall = raw_overall_summary

    result = {
        "overall": clean_overall,
        "sections": section_summaries,
    }

    result = strip_meta_from_summary(result)
    cache.set(doc_id, "summary", result)
    return result
