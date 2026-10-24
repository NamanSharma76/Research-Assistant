import json
import time
from typing import Dict, Any
from app.config import (
    GROQ_MODEL_MAIN,
    LLM_CALL_DELAY_SECONDS,
)
from app.llm import generate_json
from app.agents.schemas import Outline
from app.agents import cache
from app.agents.summarizer import summarize_document
from app.agents.notes import generate_notes
from app.agents.prompts import OUTLINE_PROMPT


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

    summary_text = summary_data.get("overall", "")
    notes_text = json.dumps(notes_data, indent=2)

    prompt = OUTLINE_PROMPT.format(
        summary=summary_text,
        notes=notes_text,
        num_slides=num_slides,
    )

    data = generate_json(prompt, schema=Outline, model=GROQ_MODEL_MAIN)
    time.sleep(LLM_CALL_DELAY_SECONDS)

    slides = data.get("slides", [])
    for idx, slide in enumerate(slides, start=1):
        slide["slide_number"] = idx
        # Build Slide.source in code from notes sources for fields in source_fields
        field_names = slide.get("source_fields", [])
        collected_citations: list[str] = []
        for fn in field_names:
            field_data = notes_data.get(fn, {})
            if isinstance(field_data, dict):
                srcs = field_data.get("sources", [])
                for s in srcs:
                    if s and s not in collected_citations:
                        collected_citations.append(s)
        slide["source"] = ", ".join(collected_citations[:2]) if collected_citations else ""

    data["slides"] = slides[:num_slides]
    validated_outline = Outline.model_validate(data)
    result = validated_outline.model_dump()

    cache.set(doc_id, cache_kind, result)
    return result
