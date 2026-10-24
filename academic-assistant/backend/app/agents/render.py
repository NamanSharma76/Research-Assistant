"""Rendering and formatting utilities for study notes, outlines, and summaries."""
from typing import Dict, Any


def render_summary(result: Dict[str, Any], doc_id: str) -> str:
    """Formats document summary as a printable markdown string."""
    lines = [
        f"=== EXECUTIVE SUMMARY ({doc_id}) ===\n",
        result.get("overall", ""),
        "\n=== SECTION SUMMARIES ===",
    ]
    for sec, text in result.get("sections", {}).items():
        lines.append(f"\n### {sec}\n{text}")
    return "\n".join(lines)


def render_notes(notes: Dict[str, Any], doc_id: str) -> str:
    """Formats structured study notes as a printable markdown string."""
    lines = [f"=== STUDY NOTES ({doc_id}) ==="]
    for field in ["problem", "method", "dataset_and_setup", "key_results", "limitations"]:
        f_data = notes.get(field, {})
        title = field.replace("_", " ").title()
        lines.append(f"\n## {title}")
        lines.append(f_data.get("text", ""))
        sources = f_data.get("sources", [])
        if sources:
            lines.append(f"Sources: {', '.join(sources)}")

    lines.append("\n## Key Terms")
    for item in notes.get("key_terms", []):
        lines.append(f"- **{item.get('term')}**: {item.get('definition')}")

    lines.append("\n## Revision Questions")
    for idx, q in enumerate(notes.get("revision_questions", []), 1):
        lines.append(f"{idx}. {q}")

    return "\n".join(lines)


def render_outline(outline: Dict[str, Any], num_slides: int) -> str:
    """Formats presentation outline as a printable string."""
    lines = [f"=== PRESENTATION OUTLINE: {outline.get('title')} ({num_slides} Slides) ==="]
    for slide in outline.get("slides", []):
        lines.append(f"\nSlide {slide.get('slide_number')}: {slide.get('title')}")
        if slide.get("source"):
            lines.append(f"Source: {slide.get('source')}")
        for b in slide.get("bullets", []):
            lines.append(f"  * {b}")
    return "\n".join(lines)
