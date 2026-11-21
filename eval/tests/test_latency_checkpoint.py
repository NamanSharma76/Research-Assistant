"""Unit test verifying latency benchmark checkpoint and resume behavior offline."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

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

from eval.latency_benchmark import (
    load_checkpoint,
    save_checkpoint,
    run_latency_benchmark,
    CHECKPOINT_FILE,
    RECOVERED_OPERATIONS,
)


def test_recovered_operations_provenance():
    assert len(RECOVERED_OPERATIONS) == 4
    for op in RECOVERED_OPERATIONS:
        assert op["recovered"] is True
        assert op["source"] == "recovered_execution_log"
        assert op["source_log"] == "task-1830.log"
        assert op["elapsed_seconds"] > 0
        assert op["llm_calls"] >= 1
        assert op["status"] == "VALID"


def test_resume_skips_recovered_and_completes_offline(tmp_path):
    from app.agents import cache
    cache.set("bdfaa68d8984", "outline_8", {"title": "Mock Outline", "slides": []})

    def mock_ask(self, query):
        mock_res = MagicMock()
        if "presentation" in query:
            mock_res.intent = "outline"
            mock_res.trace = ["router_node", "outline_node", "update_memory"]
            mock_res.llm_calls = 1
            mock_res.answer = "Sample slide outline [3 Model, p. 2]"
        elif "explain that more simply" in query:
            mock_res.intent = "rephrase_previous"
            mock_res.trace = ["router_node", "rephrase_node", "update_memory"]
            mock_res.llm_calls = 1
            mock_res.answer = "Self-attention links tokens simply."
        elif "capital of France" in query:
            mock_res.intent = "off_topic"
            mock_res.trace = ["router_node", "offtopic_node", "update_memory"]
            mock_res.llm_calls = 0
            mock_res.answer = "I am an academic research assistant."
        else:
            mock_res.intent = "paper_question"
            mock_res.trace = ["router_node", "qa_retrieve", "qa_answer", "update_memory"]
            mock_res.llm_calls = 1
            mock_res.answer = "Answer [1 Intro, p. 1]"
        return mock_res

    with patch("app.agents.graph.ChatSession.ask", new=mock_ask), \
         patch("time.sleep", return_value=None):
        summary = run_latency_benchmark(offline=True, resume=True, save_output=False)

    assert summary["total_operations"] == 8
    assert summary["valid_count"] == 8
    assert summary["recovered_count"] == 4
    assert summary["live_count"] == 4

    ops = {o["op_key"]: o for o in summary["operations"]}
    assert len(ops) == 8
    # Recovered
    assert ops["paper_question"]["recovered"] is True
    assert ops["section_summary"]["recovered"] is True
    assert ops["notes_cold"]["recovered"] is True
    assert ops["notes_warm"]["recovered"] is True
    # Newly executed
    assert ops["outline_cold"]["recovered"] is False
    assert ops["outline_warm"]["recovered"] is False
    assert ops["rephrase_previous"]["recovered"] is False
    assert ops["off_topic"]["recovered"] is False
