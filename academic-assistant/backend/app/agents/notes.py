import time
import re
from typing import Dict, Any, List
from app.config import (
    GROQ_MODEL_MAIN,
    GROQ_MODEL_FAST,
    RELEVANCE_THRESHOLD,
    LLM_CALL_DELAY_SECONDS,
)
from app.llm import generate, generate_json
from app.rag.retrieval import search, get_all_chunks
from app.agents.schemas import Notes, NoteField, Term
from app.agents import cache
from app.agents.prompts import (
    NOTE_FIELD_PROMPT,
    KEY_TERMS_PROMPT,
    REVISION_QUESTIONS_PROMPT,
)
from app.agents.cleaning import strip_meta_from_notes

# Generic search queries without paper-specific terms
NOTE_FIELD_CONFIGS = [
    {
        "field": "problem",
        "description": "Problem statement, core motivation, bottlenecks of existing methods, and research challenge addressed.",
        "query": "problem statement motivation challenge bottlenecks limitations of existing methods",
    },
    {
        "field": "method",
        "description": "Proposed model, architecture, algorithmic design, and methodological approach.",
        "query": "proposed model architecture method methodology algorithm approach framework",
    },
    {
        "field": "dataset_and_setup",
        "description": "Datasets, benchmarks, hardware, optimizer, training schedule, and experimental setup.",
        "query": "datasets benchmarks training hardware experimental setup parameters evaluation protocol",
    },
    {
        "field": "key_results",
        "description": "Key empirical results, benchmark scores, performance metrics, and ablation studies.",
        "query": "results performance evaluation metrics findings comparative results ablation study",
    },
    {
        "field": "limitations",
        "description": "Limitations, trade-offs, constraints, and future work discussed by the authors.",
        "query": "limitations trade-offs constraints threats to validity future work discussion",
    },
]

# Section title fallback targets matching Fix 1 requirements
SECTION_FALLBACK_MAP = {
    "problem": ["abstract", "introduction", "background", "motivation", "problem"],
    "method": ["method", "methods", "methodology", "approach", "model", "architecture", "framework", "proposed"],
    "dataset_and_setup": ["data", "dataset", "datasets", "setup", "evaluation protocol", "training", "experiment", "experiments", "experimental setup"],
    "key_results": ["results", "evaluation", "experiments", "findings", "performance", "ablation"],
    "limitations": ["limitations", "discussion", "threats to validity", "conclusion", "future work"],
}


def _normalize_section_title(name: str) -> str:
    """Strips leading section numbering (e.g. '1 ', '3.1 ') and lowercase for matching."""
    stripped = re.sub(r"^\d+(\.\d+)*\s*", "", (name or "").strip()).strip().lower()
    return stripped


def _find_fallback_chunks(all_chunks: List[dict], field_name: str) -> List[dict]:
    """Finds chunks belonging to sections whose normalized title matches fallback targets for field_name."""
    targets = SECTION_FALLBACK_MAP.get(field_name, [])
    if not targets:
        return []
    matched = []
    for c in all_chunks:
        sec = c.get("section", "")
        sec_norm = _normalize_section_title(sec)
        if any(t in sec_norm for t in targets):
            matched.append(c)
    return matched


def _is_not_found(text: str) -> bool:
    """Checks if LLM response indicates that the information was not found in the excerpts."""
    if not text:
        return True
    lower = text.strip().lower()
    if lower == "not found in the paper." or lower.startswith("not found"):
        return True
    not_found_phrases = [
        "not found in the paper",
        "not found in the provided excerpts",
        "do not contain enough information",
        "do not contain information",
        "does not appear to cover this",
        "no information provided",
        "no details provided in the excerpts",
        "not mentioned in the excerpts",
    ]
    return any(p in lower for p in not_found_phrases) and len(text.strip().split()) < 25


def _format_chunk_source(chunk: dict) -> str:
    sec = chunk.get("section", "Unknown")
    p_start = chunk.get("page_start", 0)
    p_end = chunk.get("page_end", 0)
    p_str = f"p. {p_start}" if p_start == p_end else f"pp. {p_start}-{p_end}"
    return f"[{sec}, {p_str}]"


def _format_excerpt_block(chunk: dict) -> str:
    src = _format_chunk_source(chunk)
    return f"{src}\n{chunk.get('text', '')}"


def generate_notes(doc_id: str, force: bool = False) -> Dict[str, Any]:
    """Generate structured study notes for a research paper.
    Caches result to data/cache/<doc_id>/notes.json.
    """
    if not force:
        cached = cache.get(doc_id, "notes")
        if cached is not None:
            return cached

    notes_dict: Dict[str, Any] = {}
    all_sent_chunks_for_terms: List[dict] = []
    all_chunks_cache: List[dict] = []

    # Process each core note field
    for cfg in NOTE_FIELD_CONFIGS:
        field_name = cfg["field"]
        chunks = search(doc_id, cfg["query"], k=4)
        
        # Check if targeted search was below threshold or empty
        best_score = chunks[0].get("score", 0.0) if chunks else 0.0
        used_fallback = False
        field_text = ""
        effective_chunks = []

        if chunks and best_score >= RELEVANCE_THRESHOLD:
            # Try targeted search chunks first
            sources = [_format_chunk_source(c) for c in chunks]
            excerpts_text = "\n\n".join(_format_excerpt_block(c) for c in chunks)
            prompt = NOTE_FIELD_PROMPT.format(
                field_name=field_name,
                field_description=cfg["description"],
                excerpts=excerpts_text,
            )
            field_text = generate(prompt, model=GROQ_MODEL_FAST)
            time.sleep(LLM_CALL_DELAY_SECONDS)

            if not _is_not_found(field_text):
                effective_chunks = chunks
            else:
                used_fallback = True
        else:
            used_fallback = True

        # Fallback to matching sections if needed
        if used_fallback and field_name in SECTION_FALLBACK_MAP:
            if not all_chunks_cache:
                all_chunks_cache = get_all_chunks(doc_id)
            fb_chunks = _find_fallback_chunks(all_chunks_cache, field_name)
            if fb_chunks:
                # Use up to 4 fallback chunks
                fb_chunks_to_use = fb_chunks[:4]
                excerpts_text = "\n\n".join(_format_excerpt_block(c) for c in fb_chunks_to_use)
                prompt = NOTE_FIELD_PROMPT.format(
                    field_name=field_name,
                    field_description=cfg["description"],
                    excerpts=excerpts_text,
                )
                fb_text = generate(prompt, model=GROQ_MODEL_FAST)
                time.sleep(LLM_CALL_DELAY_SECONDS)

                if not _is_not_found(fb_text):
                    field_text = fb_text
                    effective_chunks = fb_chunks_to_use

        if effective_chunks and field_text and not _is_not_found(field_text):
            # Sources built strictly from chunks actually used/sent to LLM
            sources = [_format_chunk_source(c) for c in effective_chunks]
            notes_dict[field_name] = {
                "text": field_text.strip(),
                "sources": sources,
            }
            all_sent_chunks_for_terms.extend(effective_chunks)
        else:
            notes_dict[field_name] = {
                "text": "Not found in the paper.",
                "sources": [],
            }

    # Dedup retrieved chunks by text/id for key terms & revision questions
    seen_ids = set()
    deduped_chunks = []
    for c in all_sent_chunks_for_terms:
        cid = c.get("id") or c.get("text", "")[:50]
        if cid not in seen_ids:
            seen_ids.add(cid)
            deduped_chunks.append(c)

    # If no chunks were sent (e.g. all empty), fallback to first few content chunks
    if not deduped_chunks:
        if not all_chunks_cache:
            all_chunks_cache = get_all_chunks(doc_id)
        deduped_chunks = [c for c in all_chunks_cache if c.get("section", "").lower() != "references"][:6]

    # Ensure combined excerpts stay within TPM token budget (use top 4 deduped chunks)
    combined_excerpts = "\n\n".join(_format_excerpt_block(c) for c in deduped_chunks[:4])

    # Key terms extraction
    terms_prompt = KEY_TERMS_PROMPT.format(excerpts=combined_excerpts)
    terms_data = generate_json(terms_prompt, model=GROQ_MODEL_FAST)
    time.sleep(LLM_CALL_DELAY_SECONDS)
    notes_dict["key_terms"] = terms_data.get("key_terms", [])

    # Revision questions extraction
    questions_prompt = REVISION_QUESTIONS_PROMPT.format(excerpts=combined_excerpts)
    questions_data = generate_json(questions_prompt, model=GROQ_MODEL_FAST)
    time.sleep(LLM_CALL_DELAY_SECONDS)
    notes_dict["revision_questions"] = questions_data.get("revision_questions", [])

    # Strip meta commentary before validation and caching
    notes_dict = strip_meta_from_notes(notes_dict)

    # Validate against Pydantic schema
    validated_notes = Notes.model_validate(notes_dict)
    result = validated_notes.model_dump()

    cache.set(doc_id, "notes", result)
    return result
