import time
from typing import Dict, Any, List
from app.config import (
    GROQ_MODEL_MAIN,
    GROQ_MODEL_FAST,
    RELEVANCE_THRESHOLD,
    LLM_CALL_DELAY_SECONDS,
)
from app.llm import generate, generate_json
from app.rag.retrieval import search
from app.agents.schemas import Notes, NoteField, Term
from app.agents import cache
from app.agents.prompts import (
    NOTE_FIELD_PROMPT,
    KEY_TERMS_PROMPT,
    REVISION_QUESTIONS_PROMPT,
)

NOTE_FIELD_CONFIGS = [
    {
        "field": "problem",
        "description": "Problem statement, core motivation, bottlenecks of existing methods, and research challenge addressed.",
        "query": "problem statement motivation challenge addressed recurrent models sequential computation limitation",
    },
    {
        "field": "method",
        "description": "Proposed model, architecture, self-attention mechanism, multi-head attention, and positional encoding.",
        "query": "Transformer model architecture Multi-Head Attention Scaled Dot-Product positional encoding feed-forward",
    },
    {
        "field": "dataset_and_setup",
        "description": "Datasets, benchmarks, hardware, optimizer, training schedule, and experimental setup.",
        "query": "datasets WMT English-German English-French training hardware GPUs optimizer Adam learning rate schedule",
    },
    {
        "field": "key_results",
        "description": "Key empirical results, benchmark scores, BLEU scores, training cost comparison, and ablation studies.",
        "query": "results performance BLEU score translation quality training cost FLOPs English-to-German English-to-French",
    },
    {
        "field": "limitations",
        "description": "Limitations, trade-offs, constraints, and future work discussed by the authors.",
        "query": "limitations future work attention restricted local context non-text modalities",
    },
]


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
    all_retrieved_chunks_for_terms: List[dict] = []

    # Process each core note field
    for cfg in NOTE_FIELD_CONFIGS:
        field_name = cfg["field"]
        chunks = search(doc_id, cfg["query"], k=4)

        if not chunks or chunks[0].get("score", 0.0) < RELEVANCE_THRESHOLD:
            notes_dict[field_name] = {
                "text": "Not found in the paper.",
                "sources": [],
            }
            continue

        # Keep chunks for terms/questions
        all_retrieved_chunks_for_terms.extend(chunks)

        # Build sources directly in Python from chunk metadata
        sources = [_format_chunk_source(c) for c in chunks]

        excerpts_text = "\n\n".join(_format_excerpt_block(c) for c in chunks)
        prompt = NOTE_FIELD_PROMPT.format(
            field_name=field_name,
            field_description=cfg["description"],
            excerpts=excerpts_text,
        )

        field_text = generate(prompt, model=GROQ_MODEL_FAST)
        time.sleep(LLM_CALL_DELAY_SECONDS)

        notes_dict[field_name] = {
            "text": field_text.strip(),
            "sources": sources,
        }

    # Dedup retrieved chunks by text/id for key terms & revision questions
    seen_ids = set()
    deduped_chunks = []
    for c in all_retrieved_chunks_for_terms:
        cid = c.get("id") or c.get("text", "")[:50]
        if cid not in seen_ids:
            seen_ids.add(cid)
            deduped_chunks.append(c)

    combined_excerpts = "\n\n".join(_format_excerpt_block(c) for c in deduped_chunks[:8])

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

    # Validate against Pydantic schema
    validated_notes = Notes.model_validate(notes_dict)
    result = validated_notes.model_dump()

    cache.set(doc_id, "notes", result)
    return result
