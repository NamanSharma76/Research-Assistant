"""Evaluates Router accuracy across 30 labeled test cases using live FAST model calls."""
import json
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.agents.router import route
from app.rag.retrieval import list_doc_sections
from app.config import LLM_CALL_DELAY_SECONDS


def main():
    doc_id = "bdfaa68d8984"
    doc_sections = list_doc_sections(doc_id)
    if not doc_sections:
        doc_sections = [
            "Unknown",
            "Abstract",
            "1 Introduction",
            "2 Background",
            "3 Model Architecture",
            "4 Why Self-Attention",
            "5 Training",
            "6 Results",
            "7 Conclusion",
            "Acknowledgements",
            "References",
        ]

    eval_file = backend_dir / "tests" / "data" / "router_eval.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"=== ROUTER EVALUATION ({len(test_cases)} cases) ===")
    print(f"Target Doc ID: {doc_id} | Sections: {len(doc_sections)}\n")

    correct = 0
    total = len(test_cases)
    mistakes = []

    print(f"{'ID':<3} | {'Expected':<18} | {'Predicted':<18} | {'Match':<5} | {'Message'}")
    print("-" * 80)

    for case in test_cases:
        cid = case["id"]
        expected = case["expected_intent"]
        msg = case["message"]
        history = case.get("history", [])
        kind = case.get("last_output_kind", "none")
        last_out = {"kind": kind, "text": "Sample text from paper"} if kind != "none" else None

        decision = route(
            user_message=msg,
            history=history,
            last_output=last_out,
            doc_sections=doc_sections,
        )

        pred = decision.intent
        is_match = (pred == expected)
        if is_match:
            correct += 1
            status = "OK"
        else:
            status = "FAIL"
            mistakes.append({
                "id": cid,
                "message": msg,
                "expected": expected,
                "predicted": pred,
                "standalone": decision.standalone_query,
                "reason": decision.reason,
            })

        print(f"{cid:<3} | {expected:<18} | {pred:<18} | {status:<5} | {msg[:35]}")
        time.sleep(LLM_CALL_DELAY_SECONDS)

    accuracy = (correct / total) * 100.0
    print("\n" + "=" * 80)
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print("=" * 80)

    if mistakes:
        print("\n=== CONFUSION / MISTAKES SUMMARY ===")
        for m in mistakes:
            print(f"- [Case #{m['id']}]: '{m['message']}'")
            print(f"    Expected : {m['expected']}")
            print(f"    Predicted: {m['predicted']}")
            print(f"    Reason   : {m['reason']}")
            print(f"    Rewritten: {m['standalone']}\n")
    else:
        print("\nAll 30 test cases passed perfectly!")


if __name__ == "__main__":
    main()
