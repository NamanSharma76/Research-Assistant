"""Unit tests for latency benchmark workflow contract validation and checkpointing."""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

eval_dir = Path(__file__).resolve().parent.parent
eval_data_dir = (eval_dir / "data").resolve()
if "DATA_DIR" not in os.environ:
    os.environ["DATA_DIR"] = str(eval_data_dir)

repo_dir = eval_dir.parent
backend_dir = repo_dir / "academic-assistant" / "backend"

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from app.agents import cache
from eval.latency_benchmark import (
    validate_operation_record,
    OPERATION_CONTRACTS,
    run_latency_benchmark,
    RECOVERED_OPERATIONS,
)


def test_operation_contracts_completeness():
    expected_keys = [
        "paper_question",
        "section_summary",
        "notes_cold",
        "notes_warm",
        "outline_cold",
        "outline_warm",
        "rephrase_previous",
        "off_topic",
    ]
    for k in expected_keys:
        assert k in OPERATION_CONTRACTS
        c = OPERATION_CONTRACTS[k]
        assert "expected_intent" in c
        assert "required_node" in c
        assert "disallowed_nodes" in c
        assert "qa_refuse" in c["disallowed_nodes"]


def test_recovered_operations_all_valid():
    for op in RECOVERED_OPERATIONS:
        val = validate_operation_record(op)
        assert val["is_valid"] is True, f"{op['op_key']} failed validation: {val['violations']}"
        assert val["status"] == "VALID"
        assert len(val["violations"]) == 0


def test_validation_detects_misrouted_operation():
    invalid_op = {
        "operation": "6. Warm-Cache 8-Slide Outline",
        "op_key": "outline_warm",
        "query": "make an 8 slide presentation",
        "intent": "paper_question",
        "llm_calls": 1,
        "elapsed_seconds": 130.609,
        "cache_hit": True,
        "trace": [
            "router_node",
            "qa_retrieve",
            "rewrite_query",
            "qa_retrieve",
            "qa_refuse",
            "update_memory",
        ],
        "answer_preview": "The paper does not appear to cover this....",
    }
    val = validate_operation_record(invalid_op)
    assert val["is_valid"] is False
    assert val["status"] == "INVALID"
    violation_str = " ".join(val["violations"])
    assert "Intent mismatch" in violation_str
    assert "Missing required graph node 'outline_node'" in violation_str
    assert "Disallowed node 'qa_refuse'" in violation_str


def test_validation_detects_cache_mismatch():
    op = {
        "operation": "4. Warm-Cache Study Notes",
        "op_key": "notes_warm",
        "query": "make study notes",
        "intent": "notes",
        "llm_calls": 1,
        "elapsed_seconds": 2.71,
        "cache_hit": False,  # Should have been True for warm
        "trace": ["router_node", "notes_node", "update_memory"],
        "answer_preview": "Valid study notes content...",
    }
    val = validate_operation_record(op)
    assert val["is_valid"] is False
    assert any("Cache behavior mismatch" in v for v in val["violations"])


def test_validation_detects_empty_answer():
    op = {
        "operation": "8. Off-Topic Query",
        "op_key": "off_topic",
        "query": "what is the capital of France?",
        "intent": "off_topic",
        "llm_calls": 0,
        "elapsed_seconds": 0.05,
        "cache_hit": None,
        "trace": ["router_node", "offtopic_node", "update_memory"],
        "answer_preview": "",  # Empty!
    }
    val = validate_operation_record(op)
    assert val["is_valid"] is False
    assert any("Empty output response" in v for v in val["violations"])


def test_mocked_offline_benchmark_run_handles_invalid_and_valid():
    cache.set("bdfaa68d8984", "outline_8", {"title": "Mock Outline", "slides": []})

    def mock_ask(self, query):
        mock_res = MagicMock()
        if "presentation" in query:
            mock_res.intent = "outline"
            mock_res.trace = ["router_node", "outline_node", "update_memory"]
            mock_res.llm_calls = 1
            mock_res.answer = "Slide 1: Title\nSlide 2: Overview"
            cache.set("bdfaa68d8984", "outline_8", {"title": "Mock Outline", "slides": []})
        elif "explain that more simply" in query:
            mock_res.intent = "rephrase_previous"
            mock_res.trace = ["router_node", "rephrase_node", "update_memory"]
            mock_res.llm_calls = 1
            mock_res.answer = "In simpler terms, self-attention connects different parts of a sentence."
        elif "capital of France" in query:
            mock_res.intent = "off_topic"
            mock_res.trace = ["router_node", "offtopic_node", "update_memory"]
            mock_res.llm_calls = 0
            mock_res.answer = "I am an academic research assistant focused on technical papers."
        else:
            mock_res.intent = "paper_question"
            mock_res.trace = ["router_node", "qa_retrieve", "qa_answer", "update_memory"]
            mock_res.llm_calls = 2
            mock_res.answer = "Self-attention connects tokens [3 Architecture, p. 3]."
        return mock_res

    with patch("app.agents.graph.ChatSession.ask", new=mock_ask), \
         patch("eval.latency_benchmark.save_checkpoint"), \
         patch("time.sleep", return_value=None):
        summary = run_latency_benchmark(offline=True, resume=True, save_output=False)

    assert summary["valid_count"] == 8
    for op in summary["operations"]:
        assert op["status"] == "VALID"
        assert len(op["violations"]) == 0
