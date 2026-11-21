import re
from typing import List, Dict, Any

META_COMMENTARY_PATTERNS = [
    # "(Field: key_results)" or "(field: problem)" or "Field: key_results"
    r"^\s*\(?\s*field\s*:\s*[a-z0-9_]+\s*\)?\s*$",
    # "All numbers are taken directly from the provided excerpts..."
    r"^\s*\*?all\s+numbers\s+are\s+taken\s+directly\s+from\s+(?:the\s+)?(?:provided\s+)?excerpts.*?\*?\s*$",
    # "All facts/details above are taken directly from the provided excerpts..."
    r"^\s*\*?all\s+(?:facts|details|information|points)\s+(?:above\s+)?are\s+taken\s+directly\s+from\s+(?:the\s+)?(?:provided\s+)?excerpts.*?\*?\s*$",
    # "...no additional information or speculation has been added..."
    r"^\s*\*?.*?no\s+additional\s+information\s+(?:or\s+speculation\s+)?(?:has\s+been|was)\s+added.*?\*?\s*$",
    # "...no additional information is inferred or added..."
    r"^\s*\*?.*?no\s+additional\s+information\s+(?:is|has\s+been)\s+(?:inferred|assumed)\s+or\s+added.*?\*?\s*$",
    # "No BLEU scores or other language-generation metrics are reported in the provided excerpts."
    r"^\s*\*?no\s+bleu\s+scores?\s+or\s+other\s+language.*?\*?\s*$",
    # "The excerpts do not contain/provide..."
    r"^\s*\*?the\s+provided\s+excerpts\s+do\s+not\s+(?:contain|provide|mention).*?\*?\s*$",
    # "If additional architectural details... they are not present in the provided excerpts."
    r"^\s*\*?if\s+additional\s+.*?are\s+not\s+present\s+in\s+the\s+provided\s+excerpts.*?\*?\s*$",
    # "Based on the provided excerpts..." / "According to the excerpts provided..." as standalone lines
    r"^\s*\*?(?:based\s+on|according\s+to)\s+the\s+(?:provided\s+)?excerpts.*?\*?\s*$",
]

COMPILED_META_PATTERNS = [re.compile(p, re.IGNORECASE) for p in META_COMMENTARY_PATTERNS]


def strip_meta_commentary(text: str) -> str:
    """Strips meta commentary, field markers, and statements about excerpts/output itself."""
    if not text:
        return ""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        # Check against meta commentary patterns
        is_meta = False
        for pattern in COMPILED_META_PATTERNS:
            if pattern.match(stripped):
                is_meta = True
                break
        if not is_meta:
            cleaned_lines.append(line)
            
    # Collapse multiple blank lines
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def strip_meta_from_notes(notes: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively or field-wise strip meta commentary from generated notes dictionary."""
    if not isinstance(notes, dict):
        return notes
    for field in ["problem", "method", "dataset_and_setup", "key_results", "limitations"]:
        if field in notes and isinstance(notes[field], dict):
            if "text" in notes[field] and isinstance(notes[field]["text"], str):
                notes[field]["text"] = strip_meta_commentary(notes[field]["text"])
        elif field in notes and isinstance(notes[field], str):
            notes[field] = strip_meta_commentary(notes[field])
    return notes


def strip_meta_from_outline(outline: Dict[str, Any]) -> Dict[str, Any]:
    """Strip meta commentary from slide bullets and titles."""
    if not isinstance(outline, dict):
        return outline
    for slide in outline.get("slides", []):
        if isinstance(slide, dict):
            bullets = slide.get("bullets", [])
            cleaned_bullets = []
            for b in bullets:
                cb = strip_meta_commentary(str(b))
                if cb:
                    cleaned_bullets.append(cb)
            slide["bullets"] = cleaned_bullets
    return outline


def strip_meta_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Strip meta commentary from overall summary and section summaries."""
    if not isinstance(summary, dict):
        return summary
    if "overall" in summary and isinstance(summary["overall"], str):
        summary["overall"] = strip_meta_commentary(summary["overall"])
    if "sections" in summary and isinstance(summary["sections"], dict):
        for sec, text in summary["sections"].items():
            if isinstance(text, str):
                summary["sections"][sec] = strip_meta_commentary(text)
    return summary
