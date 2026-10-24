"""Phase 3 tests for Router guards, LangGraph execution traces, retries, and caching."""
import pytest
from unittest.mock import patch, MagicMock
from app.agents.schemas import RouterDecision, Outline, Slide
from app.agents.router import route
from app.agents.graph import ChatSession, compiled_app
from app.agents import cache
from app.config import CACHE_VERSION


def test_router_guard_rephrase_with_no_last_output():
    decision_mock = {
        "intent": "rephrase_previous",
        "standalone_query": "make it simpler",
        "section": None,
        "num_slides": None,
        "confidence": 0.95,
        "reason": "user wants simpler explanation",
    }
    with patch("app.agents.router.generate_json", return_value=decision_mock):
        dec = route("make it simpler", history=[], last_output=None, doc_sections=["Abstract"])
        assert dec.intent == "paper_question"
        assert "no last_output" in dec.reason


def test_router_guard_low_confidence():
    decision_mock = {
        "intent": "summarize",
        "standalone_query": "something",
        "section": None,
        "num_slides": None,
        "confidence": 0.3,
        "reason": "not sure",
    }
    with patch("app.agents.router.generate_json", return_value=decision_mock):
        dec = route("something", history=[], last_output=None, doc_sections=["Abstract"])
        assert dec.intent == "paper_question"
        assert "low confidence" in dec.reason


def test_router_guard_invalid_json_fallback():
    with patch("app.agents.router.generate_json", side_effect=ValueError("Invalid json")):
        dec = route("any query", history=[], last_output=None, doc_sections=["Abstract"])
        assert dec.intent == "paper_question"
        assert dec.standalone_query == "any query"
        assert dec.confidence == 0.0


def test_router_guard_unknown_section():
    decision_mock = {
        "intent": "summarize",
        "standalone_query": "summarize section non_existent",
        "section": "NonExistentSection",
        "num_slides": None,
        "confidence": 0.9,
        "reason": "wants section summary",
    }
    with patch("app.agents.router.generate_json", return_value=decision_mock):
        dec = route("summarize non-existent", history=[], last_output=None, doc_sections=["1 Introduction", "6 Results"])
        assert dec.intent == "needs_clarification"
        assert "not found" in dec.reason


def test_router_guard_num_slides_clamped():
    decision_mock_low = {
        "intent": "outline",
        "standalone_query": "make slides",
        "section": None,
        "num_slides": 2,
        "confidence": 0.95,
        "reason": "presentation",
    }
    with patch("app.agents.router.generate_json", return_value=decision_mock_low):
        dec = route("make slides", history=[], last_output=None, doc_sections=[])
        assert dec.num_slides == 5

    decision_mock_high = {
        "intent": "outline",
        "standalone_query": "make slides",
        "section": None,
        "num_slides": 50,
        "confidence": 0.95,
        "reason": "presentation",
    }
    with patch("app.agents.router.generate_json", return_value=decision_mock_high):
        dec = route("make slides", history=[], last_output=None, doc_sections=[])
        assert dec.num_slides == 15


# --------------------------------------------------------------------------
# Graph Intent Trace Tests
# --------------------------------------------------------------------------

def test_graph_trace_paper_question():
    session = ChatSession("doc_test")
    fake_decision = RouterDecision(
        intent="paper_question",
        standalone_query="what dataset?",
        confidence=0.9,
        reason="fact question",
    )
    fake_chunk = {"section": "5 Training", "page_start": 7, "page_end": 8, "score": 0.85, "text": "WMT 2014"}

    with patch("app.agents.graph.list_doc_sections", return_value=["5 Training"]), \
         patch("app.agents.graph.route", return_value=fake_decision), \
         patch("app.agents.graph.search", return_value=[fake_chunk]), \
         patch("app.agents.graph.answer_question", return_value={
             "answer": "They used WMT 2014 [5 Training, pp. 7-8].",
             "raw_answer": "They used WMT 2014 [5 Training, pp. 7-8].",
             "refused": False,
             "citation_check": {"cited": [], "supported": [], "unsupported": []},
             "chunks": [fake_chunk],
         }):
        res = session.ask("What dataset was used?")
        assert res.trace == ["router_node", "qa_retrieve", "qa_answer", "update_memory"]
        assert "WMT 2014" in res.answer


def test_graph_trace_summarize():
    session = ChatSession("doc_test")
    fake_decision = RouterDecision(
        intent="summarize",
        standalone_query="summarize 6 Results",
        section="6 Results",
        confidence=0.9,
        reason="section summary",
    )
    with patch("app.agents.graph.list_doc_sections", return_value=["6 Results"]), \
         patch("app.agents.graph.route", return_value=fake_decision), \
         patch("app.agents.graph.summarize_section", return_value="Summary of section 6"):
        res = session.ask("summarize section 6")
        assert res.trace == ["router_node", "summarize_node", "update_memory"]
        assert res.answer == "Summary of section 6"


def test_graph_trace_notes():
    session = ChatSession("doc_test")
    fake_decision = RouterDecision(
        intent="notes",
        standalone_query="make study notes",
        confidence=0.9,
        reason="notes request",
    )
    fake_notes = {
        "problem": {"text": "problem text", "sources": []},
        "method": {"text": "method text", "sources": []},
        "dataset_and_setup": {"text": "dataset text", "sources": []},
        "key_results": {"text": "results text", "sources": []},
        "limitations": {"text": "limitations text", "sources": []},
        "key_terms": [],
        "revision_questions": [],
    }
    with patch("app.agents.graph.list_doc_sections", return_value=[]), \
         patch("app.agents.graph.route", return_value=fake_decision), \
         patch("app.agents.graph.generate_notes", return_value=fake_notes):
        res = session.ask("make study notes")
        assert res.trace == ["router_node", "notes_node", "update_memory"]


def test_graph_trace_outline():
    session = ChatSession("doc_test")
    fake_decision = RouterDecision(
        intent="outline",
        standalone_query="make 6 slides",
        num_slides=6,
        confidence=0.9,
        reason="outline request",
    )
    fake_outline = {
        "title": "Presentation",
        "slides": [{"slide_number": i, "title": f"S{i}", "bullets": ["b"], "source": ""} for i in range(1, 7)]
    }
    with patch("app.agents.graph.list_doc_sections", return_value=[]), \
         patch("app.agents.graph.route", return_value=fake_decision), \
         patch("app.agents.graph.generate_outline", return_value=fake_outline):
        res = session.ask("make 6 slides")
        assert res.trace == ["router_node", "outline_node", "update_memory"]


def test_graph_trace_off_topic_zero_extra_llm_calls():
    session = ChatSession("doc_test")
    fake_decision = RouterDecision(
        intent="off_topic",
        standalone_query="capital of France",
        confidence=0.98,
        reason="unrelated geography",
    )
    with patch("app.agents.graph.list_doc_sections", return_value=[]), \
         patch("app.agents.graph.route", return_value=fake_decision), \
         patch("app.agents.graph.generate") as mock_gen, \
         patch("app.agents.graph.search") as mock_search:
        res = session.ask("What is the capital of France?")
        assert res.trace == ["router_node", "offtopic_node", "update_memory"]
        assert "dedicated to analyzing the uploaded paper" in res.answer
        mock_gen.assert_not_called()
        mock_search.assert_not_called()


def test_graph_trace_needs_clarification_zero_extra_llm_calls():
    session = ChatSession("doc_test")
    fake_decision = RouterDecision(
        intent="needs_clarification",
        standalone_query="tell me about the other one",
        confidence=0.95,
        reason="ambiguous reference",
    )
    with patch("app.agents.graph.list_doc_sections", return_value=[]), \
         patch("app.agents.graph.route", return_value=fake_decision), \
         patch("app.agents.graph.generate") as mock_gen, \
         patch("app.agents.graph.search") as mock_search:
        res = session.ask("tell me about the other one")
        assert res.trace == ["router_node", "clarify_node", "update_memory"]
        mock_gen.assert_not_called()
        mock_search.assert_not_called()


def test_graph_trace_rephrase_no_search():
    session = ChatSession("doc_test")
    session.last_output = {"kind": "answer", "text": "Complex answer with [Section, p. 1]"}
    fake_decision = RouterDecision(
        intent="rephrase_previous",
        standalone_query="explain simpler",
        confidence=0.95,
        reason="simplify",
    )
    with patch("app.agents.graph.list_doc_sections", return_value=[]), \
         patch("app.agents.graph.route", return_value=fake_decision), \
         patch("app.agents.graph.generate", return_value="Simpler explanation [Section, p. 1]") as mock_gen, \
         patch("app.agents.graph.search") as mock_search, \
         patch("time.sleep"):
        res = session.ask("explain that more simply")
        assert res.trace == ["router_node", "rephrase_node", "update_memory"]
        assert "Simpler explanation" in res.answer
        mock_search.assert_not_called()


def test_graph_retry_loop_low_retrieval_recovers():
    session = ChatSession("doc_test")
    fake_decision = RouterDecision(
        intent="paper_question",
        standalone_query="vague question",
        confidence=0.9,
        reason="factual",
    )
    good_chunk = {"section": "3 Model Architecture", "page_start": 3, "page_end": 4, "score": 0.8, "text": "found it"}

    # First search returns empty (or below threshold), second search after rewrite returns good_chunk
    with patch("app.agents.graph.list_doc_sections", return_value=[]), \
         patch("app.agents.graph.route", return_value=fake_decision), \
         patch("app.agents.graph.search", side_effect=[[], [good_chunk]]), \
         patch("app.agents.graph.generate", return_value="optimized search keywords"), \
         patch("app.agents.graph.answer_question", return_value={
             "answer": "Answer after rewrite [3 Model Architecture, pp. 3-4].",
             "raw_answer": "raw",
             "refused": False,
             "citation_check": {"cited": [], "supported": [], "unsupported": []},
             "chunks": [good_chunk],
         }), \
         patch("time.sleep"):
        res = session.ask("vague question")
        assert res.trace == ["router_node", "qa_retrieve", "rewrite_query", "qa_retrieve", "qa_answer", "update_memory"]


def test_graph_retry_exhausted_qa_refuse():
    session = ChatSession("doc_test")
    fake_decision = RouterDecision(
        intent="paper_question",
        standalone_query="unanswerable question",
        confidence=0.9,
        reason="factual",
    )
    # Both initial search and post-rewrite search return nothing
    with patch("app.agents.graph.list_doc_sections", return_value=[]), \
         patch("app.agents.graph.route", return_value=fake_decision), \
         patch("app.agents.graph.search", side_effect=[[], []]), \
         patch("app.agents.graph.generate", return_value="optimized search keywords"), \
         patch("time.sleep"):
        res = session.ask("unanswerable question")
        assert res.trace == ["router_node", "qa_retrieve", "rewrite_query", "qa_retrieve", "qa_refuse", "update_memory"]
        assert "The paper does not appear to cover this." in res.answer


def test_graph_llm_refusal_triggers_rewrite():
    session = ChatSession("doc_test")
    fake_decision = RouterDecision(
        intent="paper_question",
        standalone_query="hard question",
        confidence=0.9,
        reason="factual",
    )
    chunk1 = {"section": "1 Introduction", "page_start": 1, "page_end": 1, "score": 0.6, "text": "intro"}
    chunk2 = {"section": "3 Model Architecture", "page_start": 3, "page_end": 4, "score": 0.85, "text": "arch"}

    # First answer is refusal, second answer succeeds
    refusal_qa = {"answer": "The paper does not appear to cover this.", "raw_answer": "refusal", "refused": True, "chunks": [chunk1]}
    success_qa = {"answer": "Detailed answer [3 Model Architecture, pp. 3-4].", "raw_answer": "ans", "refused": False, "citation_check": {"cited": [], "supported": [], "unsupported": []}, "chunks": [chunk2]}

    with patch("app.agents.graph.list_doc_sections", return_value=[]), \
         patch("app.agents.graph.route", return_value=fake_decision), \
         patch("app.agents.graph.search", side_effect=[[chunk1], [chunk2]]), \
         patch("app.agents.graph.answer_question", side_effect=[refusal_qa, success_qa]), \
         patch("app.agents.graph.generate", return_value="better query"), \
         patch("time.sleep"):
        res = session.ask("hard question")
        assert res.trace == ["router_node", "qa_retrieve", "qa_answer", "rewrite_query", "qa_retrieve", "qa_answer", "update_memory"]


def test_outline_sources_built_from_notes_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agents.cache.CACHE_BASE_DIR", tmp_path)
    fake_notes = {
        "problem": {"text": "prob", "sources": ["[1 Introduction, p. 1]"]},
        "method": {"text": "meth", "sources": ["[3 Model Architecture, pp. 3-4]"]},
        "dataset_and_setup": {"text": "data", "sources": ["[5 Training, p. 7]"]},
        "key_results": {"text": "res", "sources": ["[6 Results, p. 8]"]},
        "limitations": {"text": "lim", "sources": ["[7 Conclusion, p. 10]"]},
        "key_terms": [],
        "revision_questions": [],
    }
    raw_outline = {
        "title": "Presentation",
        "slides": [
            {
                "slide_number": 1,
                "title": "Method Slide",
                "bullets": ["b1"],
                "source_fields": ["method"],
                "source": "Model Text Placeholder",
            }
        ]
    }
    with patch("app.agents.outline.cache.get", side_effect=lambda doc, kind: fake_notes if kind == "notes" else {"overall": "ov", "sections": {}}), \
         patch("app.agents.outline.generate_notes", return_value=fake_notes), \
         patch("app.agents.outline.summarize_document", return_value={"overall": "ov", "sections": {}}), \
         patch("app.agents.outline.generate_json", return_value=raw_outline), \
         patch("time.sleep"):
        from app.agents.outline import generate_outline
        res = generate_outline("doc_id", num_slides=5, force=True)
        # Source must be derived from notes sources, not the model's text
        slide1 = res["slides"][0]
        assert slide1["source"] == "[3 Model Architecture, pp. 3-4]"


def test_cache_version_mismatch_is_miss(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agents.cache.CACHE_BASE_DIR", tmp_path)
    # Store old cache format / older version
    cache_path = tmp_path / "test_doc" / "test_kind.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"_cache_version": 999, "data": {"key": "val"}}, f)

    assert cache.get("test_doc", "test_kind") is None
