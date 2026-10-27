import time
from typing import Dict, List, Any
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

NON_CONTENT_SECTIONS = {"references", "acknowledgements", "unknown"}


def _format_chunk_with_header(chunk: dict) -> str:
    sec = chunk.get("section", "Unknown")
    p_start = chunk.get("page_start", 0)
    p_end = chunk.get("page_end", 0)
    p_str = f"p. {p_start}" if p_start == p_end else f"pp. {p_start}-{p_end}"
    return f"[{sec}, {p_str}]\n{chunk.get('text', '')}"


def summarize_section(doc_id: str, section_name: str) -> str:
    """Summarize a single section of a paper.
    If the text fits in MAX_PROMPT_TOKENS, one call is made.
    Otherwise, a map-reduce pattern summarizes chunks in batches and then combines them.
    Unsupported citations are verified and stripped.
    """
    chunks = get_section(doc_id, section_name)
    if not chunks:
        return f"Section '{section_name}' not found or empty."

    formatted_chunks = [_format_chunk_with_header(c) for c in chunks]
    full_text = "\n\n".join(formatted_chunks)

    prompt = SECTION_SUMMARY_PROMPT.format(section_name=section_name, text=full_text)

    if estimate_tokens(prompt) <= MAX_PROMPT_TOKENS:
        raw_summary = generate(prompt, model=GROQ_MODEL_FAST)
        time.sleep(LLM_CALL_DELAY_SECONDS)
    else:
        # Map-reduce approach
        # Group chunks into batches that fit in MAX_PROMPT_TOKENS
        batches: List[List[str]] = []
        current_batch: List[str] = []
        current_tokens = 0

        for chunk_text in formatted_chunks:
            chunk_tokens = estimate_tokens(chunk_text)
            if current_batch and (current_tokens + chunk_tokens > MAX_PROMPT_TOKENS - 500):
                batches.append(current_batch)
                current_batch = [chunk_text]
                current_tokens = chunk_tokens
            else:
                current_batch.append(chunk_text)
                current_tokens += chunk_tokens
        if current_batch:
            batches.append(current_batch)

        # Map phase
        partial_summaries: List[str] = []
        for batch in batches:
            batch_text = "\n\n".join(batch)
            map_prompt = MAP_SUMMARY_PROMPT.format(section_name=section_name, text=batch_text)
            part = generate(map_prompt, model=GROQ_MODEL_FAST)
            partial_summaries.append(part)
            time.sleep(LLM_CALL_DELAY_SECONDS)

        # Reduce phase
        combined_partials = "\n\n".join(partial_summaries)
        reduce_prompt = REDUCE_SUMMARY_PROMPT.format(section_name=section_name, text=combined_partials)
        raw_summary = generate(reduce_prompt, model=GROQ_MODEL_FAST)
        time.sleep(LLM_CALL_DELAY_SECONDS)

    # Citation verification & stripping
    verified = verify_citations(raw_summary, chunks)
    if verified["unsupported"]:
        clean_summary = strip_unsupported_citations(raw_summary, verified["unsupported"])
        return clean_summary
    return raw_summary


def summarize_document(doc_id: str, force: bool = False) -> Dict[str, Any]:
    """Summarize an entire document section by section (fast model) and overall synthesis (main model).
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

    cache.set(doc_id, "summary", result)
    return result
