"""Reusable UI components, parsing, and formatting helpers.

Completely decoupled from Streamlit rendering where possible to facilitate unit testing.
Escapes all untrusted model and user text prior to HTML generation.
"""
import html
import re
from typing import Any, Dict, List, Optional, Union
from api_client import ApiError

# Strict regex matching bracketed citations with section and page/page range:
# Supports both standard ASCII brackets [...] and Unicode lenticular brackets 【...】
# e.g., [3 Model Architecture, p. 3], [5 Training, pp. 7-8], 【Abstract, p. 1】
# Leaves non-citation brackets like [12], [WMT 2014], and [0, 1] untouched.
CITATION_REGEX = re.compile(r"[\[【]([^\]】]+?,\s*pp?\.\s*\d+(?:[–\-]\d+)?)[\]】]")


def render_citations_to_html(text: str) -> str:
    """Escapes raw text and converts academic citations into styled HTML chips.

    All user or model text is HTML-escaped before chip insertion to prevent
    HTML injection / XSS.
    """
    if not text:
        return ""

    escaped_text = html.escape(text)

    def _replace_citation(match: re.Match) -> str:
        citation_content = match.group(1)
        chip_html = (
            f'<span class="citation-chip" style="'
            f'display: inline-block; '
            f'background-color: #EEF2FF; '
            f'color: #3730A3; '
            f'padding: 2px 8px; '
            f'border-radius: 12px; '
            f'font-size: 0.85em; '
            f'font-weight: 600; '
            f'border: 1px solid #C7D2FE; '
            f'margin: 0 2px; '
            f'white-space: nowrap;'
            f'">[{citation_content}]</span>'
        )
        return chip_html

    return CITATION_REGEX.sub(_replace_citation, escaped_text)


def format_trace(trace: Union[List[str], str, None]) -> str:
    """Formats an agent execution trace that can be either a list or a string."""
    if not trace:
        return "N/A"
    if isinstance(trace, list):
        return " ➔ ".join(str(step) for step in trace)
    if isinstance(trace, str):
        return trace.strip()
    return str(trace)


def render_status_badge(status: str, error: Optional[str] = None) -> str:
    """Returns a visual emoji badge and status label for document states."""
    norm = (status or "").lower().strip()
    if norm == "ready":
        return "🟢 Ready"
    if norm == "processing":
        return "⏳ Processing"
    if norm == "failed":
        err_msg = f": {error}" if error else ""
        return f"🔴 Failed{err_msg}"
    return f"⚪ {status}"


def outline_to_markdown(outline_data: Dict[str, Any]) -> str:
    """Formats presentation outline JSON into a clean Markdown document."""
    lines: List[str] = []
    title = outline_data.get("title", "Presentation Outline")
    lines.append(f"# {title}\n")

    slides = outline_data.get("slides", [])
    if isinstance(slides, list):
        for idx, slide in enumerate(slides, start=1):
            if isinstance(slide, dict):
                slide_num = slide.get("slide_number", idx)
                slide_title = slide.get("title", f"Slide {slide_num}")
                source = slide.get("source", "")
                bullets = slide.get("bullets", []) or slide.get("points", [])
            else:
                slide_num = idx
                slide_title = f"Slide {slide_num}"
                source = ""
                bullets = [str(slide)]

            lines.append(f"## Slide {slide_num}: {slide_title}")
            if source:
                lines.append(f"**Sources:** {source}\n")

            for b in bullets:
                lines.append(f"- {b}")
            lines.append("")

    return "\n".join(lines).strip()


def outline_to_text(outline_data: Dict[str, Any]) -> str:
    """Formats presentation outline JSON into plain text."""
    lines: List[str] = []
    title = outline_data.get("title", "Presentation Outline")
    lines.append(f"{title.upper()}\n{'=' * len(title)}\n")

    slides = outline_data.get("slides", [])
    if isinstance(slides, list):
        for idx, slide in enumerate(slides, start=1):
            if isinstance(slide, dict):
                slide_num = slide.get("slide_number", idx)
                slide_title = slide.get("title", f"Slide {slide_num}")
                source = slide.get("source", "")
                bullets = slide.get("bullets", []) or slide.get("points", [])
            else:
                slide_num = idx
                slide_title = f"Slide {slide_num}"
                source = ""
                bullets = [str(slide)]

            lines.append(f"Slide {slide_num}: {slide_title}")
            if source:
                lines.append(f"Sources: {source}")

            for b in bullets:
                lines.append(f"  * {b}")
            lines.append("")

    return "\n".join(lines).strip()


def render_error_message(error: Union[ApiError, Exception, str]) -> str:
    """Normalizes error instances into friendly user messages without tracebacks."""
    if isinstance(error, ApiError):
        if getattr(error, "is_connection_error", False) or error.status_code == 504 or "Backend not reachable" in error.message:
            return "Backend not reachable. Check that the server is running"
        if error.status_code == 409:
            return "The previous message is still being processed, please wait."
        if error.status_code == 429:
            sec = error.retry_after or "a few"
            return f"Too many messages, try again in {sec} seconds."
        if error.status_code in (502, 503):
            return "The AI provider is temporarily unavailable, please retry."
        return error.message
    if isinstance(error, Exception):
        err_str = str(error)
        if "connection" in err_str.lower() or "not reachable" in err_str.lower():
            return "Backend not reachable. Check that the server is running"
        return err_str
    return str(error)
