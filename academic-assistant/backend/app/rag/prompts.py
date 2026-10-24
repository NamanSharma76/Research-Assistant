"""System and user prompts for RAG Q&A."""

QA_SYSTEM_PROMPT = (
    "You answer questions about a research paper using ONLY the numbered "
    "excerpts provided. Do not use outside knowledge, even if you know the "
    "answer. Do not add details that are not stated in the excerpts. Cite every "
    "claim as [Section, p. X] using only the exact section names and pages shown "
    "on the excerpts you were given. Never cite a section that is not among the "
    "excerpts. If the excerpts do not contain the answer, reply exactly: 'The "
    "paper does not appear to cover this.' Square brackets are reserved for "
    "citations. Write formulas and symbols on a single line in plain text or "
    "inline LaTeX, never in square brackets, and never leave a sentence "
    "introducing a formula without the formula."
)


def format_excerpt_header(index: int, section: str, page_start: int, page_end: int) -> str:
    """Formats the excerpt header showing section and page range exactly as cited.

    Example output: '[1] [6 Results, pp. 8-9]' or '[2] [Abstract, p. 1]'
    """
    p_str = f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}"
    return f"[{index}] [{section}, {p_str}]"


def is_refusal(text: str) -> bool:
    """Checks whether the text represents an LLM refusal to answer based on excerpts.

    Returns True if the normalized text (lowercase, stripped of leading/trailing
    whitespace and trailing punctuation) starts with 'the paper does not appear to cover this'.
    """
    if not text:
        return False
    clean = text.strip().lower().rstrip(".,;!?:")
    return clean.startswith("the paper does not appear to cover this")
