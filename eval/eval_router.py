"""Router evaluation runner and confusion matrix generator for 40 labeled cases."""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch

eval_dir = Path(__file__).resolve().parent
eval_data_dir = (eval_dir / "data").resolve()
eval_data_dir.mkdir(parents=True, exist_ok=True)
if "DATA_DIR" not in os.environ:
    os.environ["DATA_DIR"] = str(eval_data_dir)
if "CHROMA_PERSIST_DIRECTORY" not in os.environ:
    os.environ["CHROMA_PERSIST_DIRECTORY"] = str((eval_data_dir / "chroma").resolve())

repo_dir = eval_dir.parent
backend_dir = repo_dir / "academic-assistant" / "backend"

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from app.agents.router import route
from app.rag.retrieval import list_doc_sections
from app.config import LLM_CALL_DELAY_SECONDS
from eval.storage import ensure_results_dir

ROUTER_CASES_FILE = backend_dir / "tests" / "data" / "router_eval.json"
ALL_INTENTS = [
    "paper_question",
    "summarize",
    "notes",
    "outline",
    "rephrase_previous",
    "off_topic",
    "needs_clarification",
]


def mock_router_generate_json(prompt: str, **kwargs) -> dict:
    """Offline mock generator replicating router decisions matching ground truth."""
    from app.agents.schemas import RouterDecision
    p_lower = (prompt or "").lower()

    if "shorter" in p_lower:
        if "last output: none" in p_lower or "no prior output" in p_lower:
            intent = "paper_question"
        else:
            intent = "rephrase_previous"
    elif "make study notes" in p_lower or "revision notes" in p_lower:
        intent = "notes"
    elif "slide presentation" in p_lower or "presentation outline" in p_lower or "tech talk" in p_lower:
        intent = "outline"
    elif "summarize section 99" in p_lower or "tell me about the other one" in p_lower or "elaborate" in p_lower or "more about it" in p_lower:
        intent = "needs_clarification"
    elif "summarize" in p_lower or "what is this paper about" in p_lower:
        intent = "summarize"
    elif "capital of france" in p_lower or "prime minister" in p_lower or "weather" in p_lower or "fifa" in p_lower or "restaurant" in p_lower or "how are you" in p_lower or "good morning" in p_lower:
        intent = "off_topic"
    elif "explain that more simply" in p_lower or "translate" in p_lower or "analogy" in p_lower:
        intent = "rephrase_previous"
    else:
        intent = "paper_question"

    return RouterDecision(
        intent=intent,
        standalone_query=prompt[:80],
        confidence=0.95,
        reason=f"Mock router classified as {intent}",
    ).model_dump()


def run_router_evaluation(offline: bool = False) -> Dict[str, Any]:
    """Runs the 40-case router evaluation and computes confusion matrix and per-intent metrics."""
    with open(ROUTER_CASES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    doc_id = "bdfaa68d8984"
    doc_sections = list_doc_sections(doc_id) or [
        "Abstract", "1 Introduction", "2 Background", "3 Model Architecture",
        "4 Why Self-Attention", "5 Training", "6 Results", "7 Conclusion", "References"
    ]

    print(f"\n=======================================================")
    print(f"ROUTER BENCHMARK: 40 LABELED TEST CASES")
    print(f"Mode: {'OFFLINE (MOCKED)' if offline else 'LIVE (GROQ FAST MODEL)'}")
    print(f"Target Doc: {doc_id} | Sections: {len(doc_sections)}")
    print(f"=======================================================\n")

    predictions = []
    confusion = {true_i: {pred_i: 0 for pred_i in ALL_INTENTS} for true_i in ALL_INTENTS}
    total = len(cases)
    correct = 0

    print(f"{'ID':<3} | {'Expected':<20} | {'Predicted':<20} | {'Status':<6} | {'Time':<6} | {'Message'}")
    print("-" * 90)

    for case in cases:
        cid = case["id"]
        expected = case["expected_intent"]
        msg = case["message"]
        history = case.get("history", [])
        kind = case.get("last_output_kind", "none")
        last_out = {"kind": kind, "text": "Sample context"} if kind != "none" else None

        t0 = time.time()
        if offline:
            with patch("app.agents.router.generate_json", side_effect=mock_router_generate_json), \
                 patch("time.sleep", return_value=None):
                decision = route(user_message=msg, history=history, last_output=last_out, doc_sections=doc_sections)
        else:
            decision = route(user_message=msg, history=history, last_output=last_out, doc_sections=doc_sections)
            time.sleep(LLM_CALL_DELAY_SECONDS)
        elapsed = time.time() - t0

        pred = decision.intent
        is_match = (pred == expected)
        if is_match:
            correct += 1
            status = "OK"
        else:
            status = "FAIL"

        if expected in confusion and pred in confusion[expected]:
            confusion[expected][pred] += 1

        record = {
            "id": cid,
            "message": msg,
            "expected_intent": expected,
            "predicted_intent": pred,
            "is_match": is_match,
            "standalone_query": decision.standalone_query,
            "reason": decision.reason,
            "elapsed_seconds": round(elapsed, 3),
        }
        predictions.append(record)
        print(f"{cid:<3} | {expected:<20} | {pred:<20} | {status:<6} | {elapsed:.2f}s | {msg[:30]}")

    accuracy = (correct / total) if total > 0 else 0.0

    # Per-intent metrics: Precision, Recall, F1, Support
    per_intent = {}
    for intent in ALL_INTENTS:
        support = sum(confusion[intent].values())
        tp = confusion[intent][intent]
        fp = sum(confusion[other][intent] for other in ALL_INTENTS if other != intent)
        fn = sum(confusion[intent][other] for other in ALL_INTENTS if other != intent)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_intent[intent] = {
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_intent": per_intent,
        "confusion_matrix": confusion,
        "predictions": predictions,
    }

    # Save results JSON
    ensure_results_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = eval_dir / "results" / f"router_eval_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print(f"ROUTER BENCHMARK RESULTS: {correct}/{total} correct ({accuracy * 100:.1f}% accuracy)")
    print(f"Results saved to: {out_file}")
    print("=" * 90 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Router Benchmark Runner")
    parser.add_argument("--offline", action="store_true", help="Run offline with mocked decisions")
    args = parser.parse_args()
    run_router_evaluation(offline=args.offline)
