import sys
import threading
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from unittest.mock import patch
from app.agents.graph import ChatSession
from app.agents.schemas import RouterDecision
from app.llm import _get_stats_tracker, reset_call_count

turn_results = {}

def run_turn_in_thread(name, calls_to_simulate):
    reset_call_count()
    def mock_route_fn(*args, **kwargs):
        tracker = _get_stats_tracker()
        tracker.call_count += 1
        return RouterDecision(intent="paper_question", standalone_query="q", reason="r")

    def mock_answer_fn(*args, **kwargs):
        tracker = _get_stats_tracker()
        tracker.call_count += (calls_to_simulate - 1)
        return {"answer": "ans", "sources": [], "unsupported_citations": []}

    with patch("app.agents.graph.list_doc_sections", return_value=["1 Introduction", "6 Results"]), \
         patch("app.agents.graph.route", side_effect=mock_route_fn), \
         patch("app.agents.graph.search", return_value=[{"chunk_id": "c1", "text": "t", "section": "1", "page_start": 1, "page_end": 1}]), \
         patch("app.agents.graph.answer_question", side_effect=mock_answer_fn):
        s = ChatSession("test_doc")
        res = s.ask(f"question from {name}")
        turn_results[name] = res.llm_calls
        print(f"[{name}] calls_to_simulate={calls_to_simulate}, res.llm_calls={res.llm_calls}")

t1 = threading.Thread(target=run_turn_in_thread, args=("t1", 2))
t2 = threading.Thread(target=run_turn_in_thread, args=("t2", 4))
t1.start()
t2.start()
t1.join()
t2.join()
print("Final results:", turn_results)
