"""Citation verification utilities for RAG answers."""
import re
from typing import Any, Dict, List, Optional, Tuple

# Matches citations like [Section, p. X], [Section, pp. X-Y], [Section 1, p. 2; Section 2, pp. 3-4]
# Supports ASCII brackets and Unicode lenticular brackets 【...】
CITATION_BRACKET_REGEX = re.compile(r"[\[【]([^\]】]+)[\]】]")

# Candidate requirement: must contain a page marker matching \bpp?\.\s*\d or \bpages?\s*\d,
# or general page indicators like \bpp?\.|\bpages?\b
PAGE_CANDIDATE_REGEX = re.compile(r"\b(?:pp?\.|pages?\b)", re.IGNORECASE)

# Full parser for a valid candidate segment
PAGE_SPLIT_REGEX = re.compile(
    r"^(.*?)(?:,\s*|\s+)(?:pp?\.\s*|pages?\s*|p\.\s*)(\d+)(?:\s*(?:-|–|—|to)\s*(\d+))?$",
    re.IGNORECASE,
)

LEADING_NUMBER_REGEX = re.compile(r"^(?:(?:\d+\.)+\d*|\d+|[ivxlcdm]+\.?)\s*", re.IGNORECASE)


def normalize_text_chars(text: str) -> str:
    """Normalizes Unicode dashes and whitespace in text.

    Replaces:
    - Unicode dashes (U+2010 to U+2015, U+2212) with standard hyphen '-'
    - Non-breaking / thin / narrow spaces (U+00A0, U+2009, U+202F) with normal space ' '
    """
    if not text:
        return ""
    # Replace unicode dashes: U+2010 (hyphen), U+2011 (non-breaking hyphen), U+2012 (figure dash),
    # U+2013 (en dash), U+2014 (em dash), U+2015 (horizontal bar), U+2212 (minus sign)
    t = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", text)
    # Replace non-breaking / thin spaces: U+00A0 (NBSP), U+2009 (thin space), U+202F (narrow NBSP)
    t = re.sub(r"[\u00a0\u2009\u202f]", " ", t)
    return t


def extract_leading_number(sec: str) -> Optional[str]:
    """Extracts leading section number/numeral (e.g., '6' from '6 Results', '3.2' from '3.2 Model')."""
    clean = sec.strip()
    m = LEADING_NUMBER_REGEX.match(clean)
    if m:
        return m.group(0).strip().rstrip(".")
    return None


def strip_leading_number(sec: str) -> str:
    """Removes leading section numbering from section name."""
    clean = sec.strip()
    return LEADING_NUMBER_REGEX.sub("", clean).strip()


def normalize_title(sec: str) -> str:
    """Normalizes title text for matching (lowercase, stripped punctuation)."""
    clean = strip_leading_number(sec).lower()
    return clean.rstrip(".,;:").strip()


def parse_citations(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extracts all citation instances from text, tracking valid and unparseable citations.

    A bracketed segment is a citation candidate ONLY if it contains a page marker
    matching \bpp?\.\s*\d or \bpages?\s*\d. Brackets without page markers are ignored completely.

    Returns:
        (valid_citations, unparseable_citations)
    """
    norm_text = normalize_text_chars(text)
    valid_citations: List[Dict[str, Any]] = []
    unparseable_citations: List[Dict[str, Any]] = []

    for bracket_match in CITATION_BRACKET_REGEX.finditer(norm_text):
        inner_content = bracket_match.group(1).strip()
        raw_token = bracket_match.group(0)
        start_char, end_char = bracket_match.span()

        # Allow semi-colon separated citations inside a single bracket: e.g. [Abstract, p. 1; 7 Conclusion, p. 10]
        segments = [s.strip() for s in inner_content.split(";") if s.strip()]

        for seg in segments:
            # ONLY consider segments that have a page marker
            if PAGE_CANDIDATE_REGEX.search(seg):
                m = PAGE_SPLIT_REGEX.match(seg)
                if m:
                    sec_part = m.group(1).strip()
                    p_start = int(m.group(2))
                    p_end = int(m.group(3)) if m.group(3) else p_start

                    valid_citations.append({
                        "raw": raw_token,
                        "inner": seg,
                        "section": sec_part,
                        "page_start": p_start,
                        "page_end": p_end,
                        "span": (start_char, end_char),
                    })
                else:
                    # Has page marker but failed to parse (e.g. malformed or garbage)
                    unparseable_citations.append({
                        "raw": raw_token,
                        "inner": seg,
                        "section": "Unknown",
                        "page_start": None,
                        "page_end": None,
                        "reason": "unparseable",
                        "span": (start_char, end_char),
                    })

    return valid_citations, unparseable_citations


def sections_match(cit_section: str, chunk_section: str) -> bool:
    """Checks section match rule supporting subsection detail.

    Rule:
    - Split cit_section on commas: e.g. "3 Model Architecture, 3.5 Positional Encoding" -> parts[0] is "3 Model Architecture".
    - The citation is supported if the FIRST part matches the retrieved chunk's section:
        1. Normalized titles are equal after removing leading numbering ('6', '6.1', 'VI.') from both; OR
        2. The first part is ONLY a section number and equals the chunk's leading number.
    """
    parts = [p.strip() for p in cit_section.split(",") if p.strip()]
    if not parts:
        return False
    primary_part = parts[0]

    cit_raw = primary_part.strip()
    chunk_raw = chunk_section.strip()

    chunk_lead_num = extract_leading_number(chunk_raw)
    cit_title = normalize_title(cit_raw)
    chunk_title = normalize_title(chunk_raw)

    # Bare section number matches chunk's leading number (e.g. "3" matches "3 Model Architecture")
    if (cit_raw.isdigit() or cit_raw.lower() in ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]):
        if chunk_lead_num and cit_raw == chunk_lead_num:
            return True

    # Title equality after stripping leading numbering (e.g. "3 Model Architecture" vs "3 Model Architecture")
    if cit_title and chunk_title and cit_title == chunk_title:
        return True

    return False


def verify_citations(answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verifies that all citations in the answer are supported by retrieved chunks.

    Returns:
        Dict with keys:
            'cited': list of all parsed citation dicts
            'supported': list of supported citation dicts
            'unsupported': list of unsupported citation dicts (including unparseable ones)
    """
    valid_citations, unparseable_citations = parse_citations(answer)

    cited_list: List[Dict[str, Any]] = []
    supported_list: List[Dict[str, Any]] = []
    unsupported_list: List[Dict[str, Any]] = []

    # Add unparseable citations directly to cited and unsupported
    for unp in unparseable_citations:
        cited_list.append(unp)
        unsupported_list.append(unp)

    for cit in valid_citations:
        cited_list.append(cit)
        cit_sec = cit["section"]
        cit_start = cit["page_start"]
        cit_end = cit["page_end"]

        is_supported = False
        if cit_start is not None and cit_end is not None:
            for chk in chunks:
                chk_sec = str(chk.get("section", ""))
                chk_p_start = int(chk.get("page_start", 1))
                chk_p_end = int(chk.get("page_end", chk_p_start))

                sec_match = sections_match(cit_sec, chk_sec)
                page_overlap = max(cit_start, chk_p_start) <= min(cit_end, chk_p_end)

                if sec_match and page_overlap:
                    is_supported = True
                    break

        if is_supported:
            supported_list.append(cit)
        else:
            cit_copy = dict(cit)
            cit_copy["reason"] = "unsupported"
            unsupported_list.append(cit_copy)

    return {
        "cited": cited_list,
        "supported": supported_list,
        "unsupported": unsupported_list,
    }


def strip_unsupported_citations(text: str, unsupported: List[Dict[str, Any]]) -> str:
    """Removes unsupported citation bracket markers strictly by their character spans.

    Never touches other identical strings or math brackets elsewhere in text.
    """
    if not unsupported or not text:
        return text

    # Collect spans to remove
    spans = [item["span"] for item in unsupported if "span" in item]
    if not spans:
        return text

    # Sort spans in reverse order so character removals do not shift earlier indices
    spans.sort(key=lambda s: s[0], reverse=True)

    cleaned = text
    for start, end in spans:
        cleaned = cleaned[:start] + cleaned[end:]

    # Clean up double spaces or dangling spaces before punctuation left by removal
    cleaned = re.sub(r" +", " ", cleaned)
    cleaned = re.sub(r" \.", ".", cleaned)
    return cleaned.strip()
