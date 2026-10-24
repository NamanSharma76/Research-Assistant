import pytest
from unittest.mock import patch, MagicMock
from app.agents.schemas import Notes, NoteField, Term, Outline, Slide
from app.agents import cache
from app.agents.summarizer import summarize_section, summarize_document
from app.agents.notes import generate_notes
from app.agents.outline import generate_outline
from app.llm import generate_json


def test_generate_json_strips_fences_and_parses():
    fence_output = "```json\n{\"title\": \"Test Paper\", \"slides\": []}\n```"
    with patch("app.llm.generate", return_value=fence_output):
        res = generate_json("dummy prompt")
        assert res == {"title": "Test Paper", "slides": []}


def test_generate_json_retries_on_invalid_json():
    # First returns invalid JSON, second call returns valid JSON
    with patch("app.llm.generate", side_effect=["Not JSON at all", "{\"status\": \"ok\"}"]):
        res = generate_json("dummy prompt")
        assert res == {"status": "ok"}


def test_generate_json_raises_on_persistent_failure():
    with patch("app.llm.generate", side_effect=["Not JSON", "Still not JSON"]):
        with pytest.raises((RuntimeError, ValueError)):
            generate_json("dummy prompt")


def test_summarize_section_map_reduce_trigger(monkeypatch):
    """Ensure map-reduce is triggered only when prompt tokens exceed MAX_PROMPT_TOKENS."""
    fake_chunks = [
        {"section": "6 Results", "page_start": 8, "page_end": 8, "text": "Results chunk 1"},
        {"section": "6 Results", "page_start": 8, "page_end": 9, "text": "Results chunk 2"},
    ]
    with patch("app.agents.summarizer.get_section", return_value=fake_chunks), \
         patch("app.agents.summarizer.verify_citations", return_value={"cited": [], "supported": [], "unsupported": []}):
        
        # Scenario A: fits in MAX_PROMPT_TOKENS -> 1 call
        with patch("app.agents.summarizer.estimate_tokens", return_value=100), \
             patch("app.agents.summarizer.generate", return_value="Summary single call") as mock_gen, \
             patch("time.sleep"):
            res = summarize_section("dummy_doc", "6 Results")
            assert res == "Summary single call"
            assert mock_gen.call_count == 1

        # Scenario B: exceeds MAX_PROMPT_TOKENS -> map-reduce calls
        with patch("app.agents.summarizer.estimate_tokens", side_effect=[4000, 2000, 2000]), \
             patch("app.agents.summarizer.generate", side_effect=["Map 1", "Map 2", "Reduce final"]) as mock_gen, \
             patch("time.sleep"):
            res = summarize_section("dummy_doc", "6 Results")
            assert res == "Reduce final"
            assert mock_gen.call_count == 3


def test_summarize_document_skips_non_content_sections(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agents.cache.CACHE_BASE_DIR", tmp_path)
    fake_chunks = [
        {"section": "Unknown", "page_start": 1, "page_end": 1, "text": "Title block"},
        {"section": "Abstract", "page_start": 1, "page_end": 1, "text": "Abstract text"},
        {"section": "1 Introduction", "page_start": 1, "page_end": 2, "text": "Intro text"},
        {"section": "References", "page_start": 10, "page_end": 11, "text": "References list"},
        {"section": "Acknowledgements", "page_start": 10, "page_end": 10, "text": "Thanks to..."},
    ]
    with patch("app.agents.summarizer.get_all_chunks", return_value=fake_chunks), \
         patch("app.agents.summarizer.summarize_section", side_effect=lambda doc, sec: f"Summary of {sec}") as mock_sec, \
         patch("app.agents.summarizer.generate", return_value="Overall executive summary"), \
         patch("time.sleep"):
        
        res = summarize_document("test_doc", force=True)
        assert "overall" in res
        assert "Abstract" in res["sections"]
        assert "1 Introduction" in res["sections"]
        # Non-content sections skipped
        assert "Unknown" not in res["sections"]
        assert "References" not in res["sections"]
        assert "Acknowledgements" not in res["sections"]
        assert mock_sec.call_count == 2


def test_notes_below_threshold_zero_llm_calls(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agents.cache.CACHE_BASE_DIR", tmp_path)
    
    # Return low relevance chunks for all queries
    low_score_chunks = [{"section": "1 Introduction", "page_start": 1, "page_end": 1, "text": "irrelevant", "score": 0.15}]
    
    with patch("app.agents.notes.search", return_value=low_score_chunks), \
         patch("app.agents.notes.generate") as mock_gen, \
         patch("app.agents.notes.generate_json", side_effect=[{"key_terms": []}, {"revision_questions": []}]):
        
        notes = generate_notes("dummy_doc", force=True)
        # All 5 fields should be not found
        for f in ["problem", "method", "dataset_and_setup", "key_results", "limitations"]:
            assert notes[f]["text"] == "Not found in the paper."
            assert notes[f]["sources"] == []
        # mock_gen should not have been called for field generation
        assert mock_gen.call_count == 0


def test_notes_sources_constructed_from_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agents.cache.CACHE_BASE_DIR", tmp_path)
    
    mock_chunks = [
        {"id": "c1", "section": "3 Model Architecture", "page_start": 3, "page_end": 4, "text": "Arch details", "score": 0.85},
        {"id": "c2", "section": "5 Training", "page_start": 7, "page_end": 7, "text": "Training setup", "score": 0.75},
    ]
    with patch("app.agents.notes.search", return_value=mock_chunks), \
         patch("app.agents.notes.generate", return_value="Synthesized note text"), \
         patch("app.agents.notes.generate_json", side_effect=[
             {"key_terms": [{"term": "Attention", "definition": "Mechanism"}]},
             {"revision_questions": ["What is attention?"]}
         ]), \
         patch("time.sleep"):
        
        notes = generate_notes("dummy_doc", force=True)
        assert notes["problem"]["sources"] == ["[3 Model Architecture, pp. 3-4]", "[5 Training, p. 7]"]
        assert notes["problem"]["text"] == "Synthesized note text"


def test_outline_produces_exact_num_slides(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agents.cache.CACHE_BASE_DIR", tmp_path)
    
    fake_summary = {"overall": "Overall summary text", "sections": {}}
    fake_notes = {
        "problem": {"text": "p", "sources": []},
        "method": {"text": "m", "sources": []},
        "dataset_and_setup": {"text": "d", "sources": []},
        "key_results": {"text": "k", "sources": []},
        "limitations": {"text": "l", "sources": []},
        "key_terms": [],
        "revision_questions": [],
    }
    
    slides_8 = [
        {"slide_number": i, "title": f"Slide {i}", "bullets": ["b1", "b2"], "source": "Section"}
        for i in range(1, 9)
    ]
    outline_data = {"title": "Transformer Presentation", "slides": slides_8}

    with patch("app.agents.outline.cache.get", side_effect=lambda doc, kind: fake_summary if kind == "summary" else fake_notes), \
         patch("app.agents.outline.generate_notes", return_value=fake_notes), \
         patch("app.agents.outline.summarize_document", return_value=fake_summary), \
         patch("app.agents.outline.generate_json", return_value=outline_data), \
         patch("time.sleep"):
        
        outline = generate_outline("dummy_doc", num_slides=8, force=True)
        assert len(outline["slides"]) == 8
        assert [s["slide_number"] for s in outline["slides"]] == list(range(1, 9))


def test_cache_hit_avoids_llm_calls(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agents.cache.CACHE_BASE_DIR", tmp_path)
    
    # Pre-cache summary
    cached_payload = {"overall": "Cached summary", "sections": {"1 Intro": "Cached intro"}}
    cache.set("cached_doc", "summary", cached_payload)
    
    with patch("app.agents.summarizer.generate") as mock_gen:
        res = summarize_document("cached_doc", force=False)
        assert res == cached_payload
        mock_gen.assert_not_called()
        
        # When force=True, cache is bypassed
        with patch("app.agents.summarizer.get_all_chunks", return_value=[]):
            res_forced = summarize_document("cached_doc", force=True)
            assert res_forced["overall"] == "No document chunks found."
