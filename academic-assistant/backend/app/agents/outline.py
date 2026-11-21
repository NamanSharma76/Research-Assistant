import json
import time
from typing import Dict, Any, List
from app.config import (
    GROQ_MODEL_MAIN,
    LLM_CALL_DELAY_SECONDS,
)
from app.llm import generate_json
from app.rag.retrieval import get_all_chunks
from app.agents.schemas import Outline
from app.agents import cache
from app.agents.summarizer import summarize_document
from app.agents.notes import generate_notes
from app.agents.prompts import OUTLINE_PROMPT
from app.agents.cleaning import strip_meta_from_outline


REGEN_SLIDE_PROMPT = """You are an academic research assistant designing a slide for a presentation outline.
You must generate a single slide based strictly on the provided study notes.
The slide must draw from at least one valid source field from the notes: ["problem", "method", "dataset_and_setup", "key_results", "limitations"].

Notes:
{notes}

Slide Title: {slide_title}
Slide Number: {slide_number}

Requirements:
- Provide 3 to 5 concise bullet points.
- List the source_fields from the notes it draws on: ["problem", "method", "dataset_and_setup", "key_results", "limitations"]. Choose at least one field that has content in the notes.
- Do not state as fact anything the notes present as a hypothesis or possibility. Report it as stated.
- If notes give two different numbers for the same result, report both as stated.
- Do not add detail the notes do not state.
- Do not add praise, conclusions or editorial claims.

Return a valid JSON object matching this schema:
{{
  "slide_number": {slide_number},
  "title": "{slide_title}",
  "bullets": ["<Bullet 1>", "<Bullet 2>", "<Bullet 3>"],
  "source_fields": ["<field name>"]
}}
"""


def _get_first_page_context(doc_id: str) -> str:
    """Extracts first-page chunk text to supply title and author grounding strictly from the document."""
    try:
        chunks = get_all_chunks(doc_id)
        p1_chunks = [c for c in chunks if c.get("page_start", 0) == 1]
        if not p1_chunks and chunks:
            p1_chunks = chunks[:2]
        lines = []
        for c in p1_chunks:
            lines.append(c.get("text", ""))
        combined = "\n\n".join(lines).strip()
        return combined[:2000]
    except Exception:
        return ""


def _build_slide_source(field_names: List[str], notes_data: Dict[str, Any]) -> str:
    collected_citations: list[str] = []
    for fn in field_names:
        field_data = notes_data.get(fn, {})
        if isinstance(field_data, dict):
            srcs = field_data.get("sources", [])
            for s in srcs:
                if s and s not in collected_citations:
                    collected_citations.append(s)
    return ", ".join(collected_citations[:2]) if collected_citations else ""


def generate_outline(doc_id: str, num_slides: int = 10, force: bool = False) -> Dict[str, Any]:
    """Generate a structured presentation outline grounded in summary and notes.
    num_slides must be between 5 and 15.
    Caches result to data/cache/<doc_id>/outline_<num_slides>.json.
    """
    if not (5 <= num_slides <= 15):
        raise ValueError(f"num_slides must be between 5 and 15, got {num_slides}")

    cache_kind = f"outline_{num_slides}"
    if not force:
        cached = cache.get(doc_id, cache_kind)
        if cached is not None:
            return cached

    # Ensure document summary and notes exist
    summary_data = None if force else cache.get(doc_id, "summary")
    if summary_data is None:
        summary_data = summarize_document(doc_id, force=force)

    notes_data = None if force else cache.get(doc_id, "notes")
    if notes_data is None:
        notes_data = generate_notes(doc_id, force=force)

    summary_text = summary_data.get("overall", "")[:2500]
    # Compact notes text to fit within MAX_PROMPT_TOKENS
    compact_notes = {}
    for k in ["problem", "method", "dataset_and_setup", "key_results", "limitations"]:
        f_data = notes_data.get(k, {})
        if isinstance(f_data, dict):
            compact_notes[k] = {
                "text": f_data.get("text", "")[:1200],
                "sources": f_data.get("sources", [])[:3]
            }
        elif isinstance(f_data, str):
            compact_notes[k] = {"text": f_data[:1200], "sources": []}
    notes_text = json.dumps(compact_notes, indent=2)

    first_page_context = _get_first_page_context(doc_id)

    prompt = OUTLINE_PROMPT.format(
        first_page_context=first_page_context,
        summary=summary_text,
        notes=notes_text,
        num_slides=num_slides,
    )

    data = generate_json(prompt, schema=Outline, model=GROQ_MODEL_MAIN)
    time.sleep(LLM_CALL_DELAY_SECONDS)

    slides = data.get("slides", [])
    for idx, slide in enumerate(slides, start=1):
        slide["slide_number"] = idx
        field_names = slide.get("source_fields", [])

        # Fix 5c: If slide is a limitations slide, enforce that it only draws from limitations
        title_lower = slide.get("title", "").lower()
        if "limitation" in title_lower or "threats" in title_lower:
            field_names = ["limitations"]
            slide["source_fields"] = ["limitations"]

        slide_source = _build_slide_source(field_names, notes_data)

        # Fix 5b: If slide has no source, regenerate it once from the notes
        if not slide_source:
            regen_prompt = REGEN_SLIDE_PROMPT.format(
                notes=notes_text,
                slide_title=slide.get("title", f"Slide {idx}"),
                slide_number=idx,
            )
            try:
                regen_data = generate_json(regen_prompt, model=GROQ_MODEL_MAIN)
                time.sleep(LLM_CALL_DELAY_SECONDS)
                if isinstance(regen_data, dict):
                    new_bullets = regen_data.get("bullets", [])
                    if new_bullets:
                        slide["bullets"] = new_bullets
                    new_fields = regen_data.get("source_fields", [])
                    if new_fields:
                        slide["source_fields"] = new_fields
                        slide_source = _build_slide_source(new_fields, notes_data)
            except Exception:
                pass

            # If still has no source, mark it "(no source)" rather than leaving it blank
            if not slide_source:
                slide_source = "(no source)"

        slide["source"] = slide_source

    data["slides"] = slides[:num_slides]
    data = strip_meta_from_outline(data)
    validated_outline = Outline.model_validate(data)
    result = validated_outline.model_dump()

    cache.set(doc_id, cache_kind, result)
    return result
