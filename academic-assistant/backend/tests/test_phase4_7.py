"""Unit tests for Phase 4.7 robustness fixes across notes, outline, and summary."""
import pytest
from unittest.mock import patch, MagicMock
from app.agents.notes import generate_notes, _normalize_section_title, _find_fallback_chunks
from app.agents.outline import generate_outline
from app.agents.cleaning import strip_meta_commentary, strip_meta_from_notes, strip_meta_from_outline, strip_meta_from_summary


def test_section_title_normalization_ignores_numbering_and_case():
    assert _normalize_section_title("1 Introduction") == "introduction"
    assert _normalize_section_title("3.1 Architecture") == "architecture"
    assert _normalize_section_title("6 Discussion And Threats To Validity") == "discussion and threats to validity"
    assert _normalize_section_title("  4.2.1 Experimental Setup  ") == "experimental setup"
    assert _normalize_section_title("Abstract") == "abstract"


def test_targeted_search_below_threshold_fallback_finds_content_for_all_three_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agents.cache.CACHE_BASE_DIR", tmp_path)
    
    # 1. Search returns low score chunk (below RELEVANCE_THRESHOLD 0.25)
    low_chunks = [{"section": "Unrelated", "page_start": 5, "page_end": 5, "text": "Low score text", "score": 0.12}]
    
    # 2. get_all_chunks returns sections matching the 3 fields
    all_chunks = [
        {"section": "1 Background And Motivation", "page_start": 1, "page_end": 1, "text": "Problem details here"},
        {"section": "3 Our Approach", "page_start": 2, "page_end": 2, "text": "Methodology architecture here"},
        {"section": "6 Discussion And Threats To Validity", "page_start": 3, "page_end": 3, "text": "Limitations details here"},
    ]
    
    with patch("app.agents.notes.search", return_value=low_chunks), \
         patch("app.agents.notes.get_all_chunks", return_value=all_chunks), \
         patch("app.agents.notes.generate", return_value="Synthesized fallback field content") as mock_gen, \
         patch("app.agents.notes.generate_json", side_effect=[
             {"key_terms": [{"term": "TCN", "definition": "Network"}]},
             {"revision_questions": ["What is TCN?"]}
         ]), \
         patch("time.sleep"):
        
        notes = generate_notes("dummy_fallback_doc", force=True)
        
        # All three fields should have found content via fallback
        assert notes["problem"]["text"] == "Synthesized fallback field content"
        assert notes["problem"]["sources"] == ["[1 Background And Motivation, p. 1]"]
        
        assert notes["method"]["text"] == "Synthesized fallback field content"
        assert notes["method"]["sources"] == ["[3 Our Approach, p. 2]"]
        
        assert notes["limitations"]["text"] == "Synthesized fallback field content"
        assert notes["limitations"]["sources"] == ["[6 Discussion And Threats To Validity, p. 3]"]
        
        # LLM was invoked for the fallback chunks
        assert mock_gen.call_count >= 3


def test_meta_commentary_is_stripped_from_outputs():
    raw_text = (
        "(Field: key_results)\n"
        "The model achieves 97.8% accuracy.\n"
        "All numbers are taken directly from the provided excerpts; no additional information has been added.\n"
        "No BLEU scores or other language-generation metrics are reported in the provided excerpts.\n"
        "Training took 11 hours."
    )
    cleaned = strip_meta_commentary(raw_text)
    assert "(Field: key_results)" not in cleaned
    assert "All numbers are taken directly" not in cleaned
    assert "No BLEU scores" not in cleaned
    assert "The model achieves 97.8% accuracy." in cleaned
    assert "Training took 11 hours." in cleaned


def test_source_not_listed_when_chunks_not_sent_to_llm(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agents.cache.CACHE_BASE_DIR", tmp_path)
    
    # Suppose search returns chunk 1 and chunk 2, but for this field only chunk 1 is used
    mock_chunks = [
        {"id": "c1", "section": "4 Experimental Setup", "page_start": 2, "page_end": 2, "text": "Dataset info", "score": 0.8},
    ]
    all_chunks = [
        {"section": "2 Related Work", "page_start": 1, "page_end": 1, "text": "Related work not sent to setup field"},
        {"section": "4 Experimental Setup", "page_start": 2, "page_end": 2, "text": "Dataset info"},
    ]
    
    with patch("app.agents.notes.search", return_value=mock_chunks), \
         patch("app.agents.notes.get_all_chunks", return_value=all_chunks), \
         patch("app.agents.notes.generate", return_value="Dataset summary"), \
         patch("app.agents.notes.generate_json", side_effect=[
             {"key_terms": []},
             {"revision_questions": []}
         ]), \
         patch("time.sleep"):
        
        notes = generate_notes("dummy_doc", force=True)
        # Related Work was NOT sent to the LLM for dataset_and_setup, so it must not be in sources
        assert "[2 Related Work, p. 1]" not in notes["dataset_and_setup"]["sources"]
        assert notes["dataset_and_setup"]["sources"] == ["[4 Experimental Setup, p. 2]"]


def test_outline_built_from_chunks_with_no_author_text_does_not_contain_authors(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agents.cache.CACHE_BASE_DIR", tmp_path)
    
    # First page chunk contains only title, no author text
    first_page_chunk = [{"section": "Unknown", "page_start": 1, "page_end": 1, "text": "GridTCN: Day-Ahead Electricity Demand Forecasting"}]
    
    fake_summary = {"overall": "Executive overview", "sections": {}}
    fake_notes = {
        "problem": {"text": "p", "sources": ["[1 Background, p. 1]"]},
        "method": {"text": "m", "sources": ["[3 Method, p. 2]"]},
        "dataset_and_setup": {"text": "d", "sources": ["[4 Data, p. 2]"]},
        "key_results": {"text": "k", "sources": ["[5 Results, p. 3]"]},
        "limitations": {"text": "l", "sources": ["[6 Discussion, p. 3]"]},
        "key_terms": [],
        "revision_questions": [],
    }
    
    # Mock LLM generating outline adhering to the rule (no authors)
    slides_data = {
        "title": "GridTCN: Day-Ahead Electricity Demand Forecasting",
        "slides": [
            {
                "slide_number": 1,
                "title": "GridTCN: Day-Ahead Electricity Demand Forecasting",
                "bullets": ["Lightweight dilated TCN for electricity demand forecasting."],
                "source_fields": ["problem"]
            },
            {
                "slide_number": 2,
                "title": "Problem Statement",
                "bullets": ["Demand forecasting is critical for substation operations."],
                "source_fields": ["problem"]
            },
            {
                "slide_number": 3,
                "title": "Methodology",
                "bullets": ["Dilated causal residual blocks."],
                "source_fields": ["method"]
            },
            {
                "slide_number": 4,
                "title": "Experimental Protocol",
                "bullets": ["3 years of substation data."],
                "source_fields": ["dataset_and_setup"]
            },
            {
                "slide_number": 5,
                "title": "Results",
                "bullets": ["4.3% MAPE achieved."],
                "source_fields": ["key_results"]
            },
        ]
    }
    
    with patch("app.agents.outline.get_all_chunks", return_value=first_page_chunk), \
         patch("app.agents.outline.cache.get", return_value=None), \
         patch("app.agents.outline.summarize_document", return_value=fake_summary), \
         patch("app.agents.outline.generate_notes", return_value=fake_notes), \
         patch("app.agents.outline.generate_json", return_value=slides_data), \
         patch("time.sleep"):
        
        outline = generate_outline("no_author_doc", num_slides=5, force=True)
        # Verify slide 1 has no invented authors
        slide1_text = " ".join(outline["slides"][0]["bullets"])
        assert "Vaswani" not in slide1_text
        assert "Asha Verma" not in slide1_text
        assert "Karan Mehta" not in slide1_text
        assert "Conference" not in slide1_text


def test_slide_with_no_source_is_regenerated_once_then_marked_no_source(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agents.cache.CACHE_BASE_DIR", tmp_path)
    
    fake_summary = {"overall": "Summary", "sections": {}}
    fake_notes = {
        "problem": {"text": "p", "sources": []},  # Empty sources
        "method": {"text": "m", "sources": []},
        "dataset_and_setup": {"text": "d", "sources": []},
        "key_results": {"text": "k", "sources": []},
        "limitations": {"text": "l", "sources": []},
        "key_terms": [],
        "revision_questions": [],
    }
    
    slides_data = {
        "title": "Presentation",
        "slides": [
            {
                "slide_number": 1,
                "title": "Introduction",
                "bullets": ["Introduction bullet"],
                "source_fields": []  # No source fields
            },
            {
                "slide_number": 2,
                "title": "Motivation",
                "bullets": ["Motivation bullet"],
                "source_fields": []
            },
            {
                "slide_number": 3,
                "title": "Approach",
                "bullets": ["Approach bullet"],
                "source_fields": []
            },
            {
                "slide_number": 4,
                "title": "Results",
                "bullets": ["Results bullet"],
                "source_fields": []
            },
            {
                "slide_number": 5,
                "title": "Conclusion",
                "bullets": ["Conclusion bullet"],
                "source_fields": []
            }
        ]
    }
    
    # generate_json will be called once for main outline, then for slide regenerations
    with patch("app.agents.outline.get_all_chunks", return_value=[]), \
         patch("app.agents.outline.cache.get", return_value=None), \
         patch("app.agents.outline.summarize_document", return_value=fake_summary), \
         patch("app.agents.outline.generate_notes", return_value=fake_notes), \
         patch("app.agents.outline.generate_json", return_value=slides_data), \
         patch("time.sleep"):
        
        outline = generate_outline("dummy_doc", num_slides=5, force=True)
        # Because notes had empty sources, after regeneration it still has no source and is marked "(no source)"
        for s in outline["slides"]:
            assert s["source"] == "(no source)"
