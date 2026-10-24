"""Unit tests for citation verification logic."""
import pytest
from app.rag.citations import verify_citations, strip_unsupported_citations, parse_citations


def test_verify_citations_supported_exact_and_range():
    """Test that valid citations (single page and page range) are marked as supported."""
    chunks = [
        {"section": "6 Results", "page_start": 8, "page_end": 9},
        {"section": "Abstract", "page_start": 1, "page_end": 1},
        {"section": "5 Training", "page_start": 7, "page_end": 8},
    ]

    answer = (
        "The model achieves 28.4 BLEU on English-to-German [6 Results, pp. 8-9]. "
        "The architecture dispenses with recurrence [Abstract, p. 1]."
    )

    result = verify_citations(answer, chunks)
    assert len(result["cited"]) == 2
    assert len(result["supported"]) == 2
    assert len(result["unsupported"]) == 0
    assert result["supported"][0]["section"] == "6 Results"
    assert result["supported"][1]["section"] == "Abstract"


def test_verify_citations_unsupported_wrong_section():
    """Test that a citation with a non-existent section is marked as unsupported."""
    chunks = [
        {"section": "6 Results", "page_start": 8, "page_end": 9},
        {"section": "Abstract", "page_start": 1, "page_end": 1},
    ]

    # Cites 'Related Work' which is not in retrieved chunks
    answer = "Prior work utilized LSTM encoders [Related Work, p. 2]."

    result = verify_citations(answer, chunks)
    assert len(result["cited"]) == 1
    assert len(result["supported"]) == 0
    assert len(result["unsupported"]) == 1
    assert result["unsupported"][0]["section"] == "Related Work"


def test_verify_citations_unsupported_wrong_page():
    """Test that a citation with matching section but out-of-range page is marked as unsupported."""
    chunks = [
        {"section": "6 Results", "page_start": 8, "page_end": 9},
    ]

    # Section matches '6 Results', but page is 15 (outside 8..9)
    answer = "The ensemble model scored higher [6 Results, p. 15]."

    result = verify_citations(answer, chunks)
    assert len(result["cited"]) == 1
    assert len(result["supported"]) == 0
    assert len(result["unsupported"]) == 1
    assert result["unsupported"][0]["page_start"] == 15


def test_strip_unsupported_citations():
    """Test clean stripping of unsupported citations from answer string."""
    chunks = [
        {"section": "6 Results", "page_start": 8, "page_end": 9},
    ]
    answer = "Transformer big attained 28.4 [6 Results, pp. 8-9] but CNN models failed [Vision, p. 3]."

    result = verify_citations(answer, chunks)
    assert len(result["unsupported"]) == 1

    stripped = strip_unsupported_citations(answer, result["unsupported"])
    assert "[Vision, p. 3]" not in stripped
    assert "[6 Results, pp. 8-9]" in stripped


def test_parse_handles_unicode_dashes_and_non_breaking_spaces():
    """Test parse handles non-breaking hyphen (\\u2011), en-dash (\\u2013), and em-dash (\\u2014)."""
    text = "Achieved 28.4 BLEU [6\u00a0Results, pp.\u202f7\u20118] and [5\u2009Training, pp.\u00a07\u20138]."
    chunks = [
        {"section": "6 Results", "page_start": 7, "page_end": 8},
        {"section": "5 Training", "page_start": 7, "page_end": 8},
    ]

    result = verify_citations(text, chunks)
    assert len(result["cited"]) == 2
    assert len(result["supported"]) == 2
    assert len(result["unsupported"]) == 0


def test_unparseable_page_citation_marked_unsupported():
    """Test that an unparseable page citation ends up in unsupported with reason 'unparseable' and None pages."""
    text = "The model trained for several days [Training, pages unknown]."
    chunks = [
        {"section": "Training", "page_start": 1, "page_end": 5},
    ]

    result = verify_citations(text, chunks)
    assert len(result["cited"]) == 1
    assert len(result["supported"]) == 0
    assert len(result["unsupported"]) == 1
    assert result["unsupported"][0]["reason"] == "unparseable"
    assert result["unsupported"][0]["page_start"] is None
    assert result["unsupported"][0]["page_end"] is None


def test_bare_section_number_matches_chunk():
    """Test that bare '[3, p. 2]' matches '3 Model Architecture' pp. 2-4 but NOT '5 Training'."""
    chunks = [
        {"section": "3 Model Architecture", "page_start": 2, "page_end": 4},
        {"section": "5 Training", "page_start": 7, "page_end": 8},
    ]

    res1 = verify_citations("See architecture details [3, p. 2].", chunks)
    assert len(res1["supported"]) == 1
    assert len(res1["unsupported"]) == 0

    chunks_no_3 = [{"section": "5 Training", "page_start": 1, "page_end": 10}]
    res2 = verify_citations("See architecture details [3, p. 2].", chunks_no_3)
    assert len(res2["supported"]) == 0
    assert len(res2["unsupported"]) == 1


def test_results_title_match_ignoring_numbering():
    """Test that '[Results, p. 8]' matches '6 Results' pp. 8-9 (title match, numbering ignored)."""
    chunks = [
        {"section": "6 Results", "page_start": 8, "page_end": 9},
    ]

    result = verify_citations("Best score in table 2 [Results, p. 8].", chunks)
    assert len(result["supported"]) == 1
    assert len(result["unsupported"]) == 0


def test_bracketed_math_formula_not_treated_as_citation():
    """Test an answer containing 'PE(pos,2i) = sin(pos / 10000^(2i/dmodel))' in square brackets plus one valid citation."""
    answer = (
        "Positional encodings use formula [PE(pos,2i) = sin(pos / 10000^(2i/dmodel))] "
        "as described in [3 Model Architecture, p. 6]."
    )
    chunks = [
        {"section": "3 Model Architecture", "page_start": 5, "page_end": 6},
    ]

    result = verify_citations(answer, chunks)
    # Only the valid citation is in cited
    assert len(result["cited"]) == 1
    assert result["cited"][0]["section"] == "3 Model Architecture"
    assert len(result["supported"]) == 1
    assert len(result["unsupported"]) == 0

    # After stripping, formula text is completely unchanged
    cleaned = strip_unsupported_citations(answer, result["unsupported"])
    assert "[PE(pos,2i) = sin(pos / 10000^(2i/dmodel))]" in cleaned


def test_subsection_detail_inside_citation_is_supported():
    """Test '[3 Model Architecture, 3.5 Positional Encoding, pp. 5-6]' is supported when chunk '3 Model Architecture' pp. 5-6 exists."""
    answer = "The encoding is defined in [3 Model Architecture, 3.5 Positional Encoding, pp. 5-6]."
    chunks = [
        {"section": "3 Model Architecture", "page_start": 5, "page_end": 6},
    ]

    result = verify_citations(answer, chunks)
    assert len(result["cited"]) == 1
    assert len(result["supported"]) == 1
    assert len(result["unsupported"]) == 0


def test_results_garbage_page_marker_is_unsupported_unparseable():
    """Test '[Results, pp. x-y]' style garbage with a page marker is unsupported with reason 'unparseable' and page_start None."""
    answer = "Refer to table [Results, pp. x-y]."
    chunks = [
        {"section": "6 Results", "page_start": 8, "page_end": 9},
    ]

    result = verify_citations(answer, chunks)
    assert len(result["cited"]) == 1
    assert len(result["unsupported"]) == 1
    un = result["unsupported"][0]
    assert un["reason"] == "unparseable"
    assert un["page_start"] is None
    assert un["page_end"] is None


def test_span_stripping_preserves_identical_brackets_elsewhere():
    """Test that stripping removes only the exact character span of an unsupported citation and leaves an identical-looking bracket elsewhere intact."""
    # Two identical strings "[Bad Section, p. 99]" at different positions, but suppose only the first was flagged unsupported
    text = "First [Bad Section, p. 99] and second [Bad Section, p. 99]."
    # Create fake unsupported with span pointing only to the second occurrence
    span_second = (text.rfind("[Bad Section, p. 99]"), text.rfind("[Bad Section, p. 99]") + len("[Bad Section, p. 99]"))
    fake_unsupported = [{"raw": "[Bad Section, p. 99]", "span": span_second}]

    cleaned = strip_unsupported_citations(text, fake_unsupported)
    # The first occurrence must be intact
    assert cleaned.startswith("First [Bad Section, p. 99]")
    # The second occurrence must be stripped
    assert not cleaned.endswith("[Bad Section, p. 99].")
