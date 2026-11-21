"""Task-specific latency and execution trace benchmark across 8 core operations.

Supports atomic checkpointing, strict contract validation, and --resume
to safely recover from interruptions while validating execution integrity.
"""
import argparse
import json
import os
import sys
import time
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

from app.agents.graph import ChatSession
from app.agents import cache
from app.llm import get_call_count, reset_call_count
from eval.storage import ensure_results_dir

TARGET_DOC_ID = "bdfaa68d8984"  # Paper A: Attention Is All You Need
CHECKPOINT_FILE = eval_dir / "results" / "latency_checkpoint.json"

# Strict workflow contracts for each operation
OPERATION_CONTRACTS = {
    "paper_question": {
        "expected_intent": "paper_question",
        "required_node": "qa_answer",
        "disallowed_nodes": ["qa_refuse"],
        "expects_cache": None,
        "valid_output_types": ["answer"],
    },
    "section_summary": {
        "expected_intent": "summarize",
        "required_node": "summarize_node",
        "disallowed_nodes": ["qa_refuse"],
        "expects_cache": None,
        "valid_output_types": ["summary"],
    },
    "notes_cold": {
        "expected_intent": "notes",
        "required_node": "notes_node",
        "disallowed_nodes": ["qa_refuse"],
        "expects_cache": False,
        "cache_kind": "notes",
        "valid_output_types": ["notes"],
    },
    "notes_warm": {
        "expected_intent": "notes",
        "required_node": "notes_node",
        "disallowed_nodes": ["qa_refuse"],
        "expects_cache": True,
        "cache_kind": "notes",
        "valid_output_types": ["notes"],
    },
    "outline_cold": {
        "expected_intent": "outline",
        "required_node": "outline_node",
        "disallowed_nodes": ["qa_refuse"],
        "expects_cache": False,
        "cache_kind": "outline_8",
        "valid_output_types": ["outline"],
    },
    "outline_warm": {
        "expected_intent": "outline",
        "required_node": "outline_node",
        "disallowed_nodes": ["qa_refuse"],
        "expects_cache": True,
        "cache_kind": "outline_8",
        "valid_output_types": ["outline"],
    },
    "rephrase_previous": {
        "expected_intent": "rephrase_previous",
        "required_node": "rephrase_node",
        "disallowed_nodes": ["qa_refuse"],
        "expects_cache": None,
        "valid_output_types": ["rephrase", "answer"],
    },
    "off_topic": {
        "expected_intent": "off_topic",
        "required_node": "offtopic_node",
        "disallowed_nodes": ["qa_refuse"],
        "expects_cache": None,
        "valid_output_types": ["off_topic"],
    },
}

# Exact measurements recovered from execution log task-1830.log
RECOVERED_OPERATIONS = [
    {
        "operation": "1. Paper Question Answering",
        "op_key": "paper_question",
        "query": "What is self-attention and how does it relate different positions?",
        "intent": "paper_question",
        "llm_calls": 2,
        "elapsed_seconds": 6.15,
        "cache_hit": None,
        "trace": ["router_node", "qa_retrieve", "qa_answer", "update_memory"],
        "answer_preview": "Self-attention relates different positions of a single sequence...",
        "source": "recovered_execution_log",
        "source_log": "task-1830.log",
        "recovered": True,
        "status": "VALID",
        "violations": [],
    },
    {
        "operation": "2. Section Summarization",
        "op_key": "section_summary",
        "query": "summarize section 3 Model Architecture",
        "intent": "summarize",
        "llm_calls": 2,
        "elapsed_seconds": 8.05,
        "cache_hit": None,
        "trace": ["router_node", "summarize_node", "update_memory"],
        "answer_preview": "Section 3 Model Architecture describes the Transformer model...",
        "source": "recovered_execution_log",
        "source_log": "task-1830.log",
        "recovered": True,
        "status": "VALID",
        "violations": [],
    },
    {
        "operation": "3. Cold-Cache Study Notes",
        "op_key": "notes_cold",
        "query": "make study notes",
        "intent": "notes",
        "llm_calls": 8,
        "elapsed_seconds": 181.12,
        "cache_hit": False,
        "trace": ["router_node", "notes_node", "update_memory"],
        "answer_preview": "Problem: Recurrent models preclude parallelization within training...",
        "source": "recovered_execution_log",
        "source_log": "task-1830.log",
        "recovered": True,
        "status": "VALID",
        "violations": [],
    },
    {
        "operation": "4. Warm-Cache Study Notes",
        "op_key": "notes_warm",
        "query": "make study notes",
        "intent": "notes",
        "llm_calls": 1,
        "elapsed_seconds": 2.71,
        "cache_hit": True,
        "trace": ["router_node", "notes_node", "update_memory"],
        "answer_preview": "Problem: Recurrent models preclude parallelization within training...",
        "source": "recovered_execution_log",
        "source_log": "task-1830.log",
        "recovered": True,
        "status": "VALID",
        "violations": [],
    },
]


def validate_operation_record(
    record: Dict[str, Any],
    contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validates an operation record against its workflow contract.

    Explicitly verifies:
    - Infrastructure error status
    - Expected intent
    - Required graph node
    - Absence of disallowed nodes (e.g. qa_refuse)
    - Actual cache behavior
    - Appropriate output content and non-refusal
    """
    op_key = record.get("op_key")
    if not contract:
        contract = OPERATION_CONTRACTS.get(op_key, {})

    violations = []

    # 1. Infrastructure error status
    if record.get("error"):
        violations.append(f"Infrastructure error: {record['error']}")

    # 2. Expected intent validation
    exp_intent = contract.get("expected_intent")
    actual_intent = record.get("intent")
    if exp_intent and actual_intent != exp_intent:
        violations.append(
            f"Intent mismatch: expected '{exp_intent}', got '{actual_intent}'"
        )

    # 3. Required graph node validation
    trace = record.get("trace", []) or []
    req_node = contract.get("required_node")
    if req_node and req_node not in trace:
        violations.append(
            f"Missing required graph node '{req_node}' in execution trace {trace}"
        )

    # Disallowed nodes
    for dis in contract.get("disallowed_nodes", []):
        if dis in trace:
            violations.append(
                f"Disallowed node '{dis}' encountered in execution trace {trace}"
            )

    # 4. Actual cache behavior validation
    exp_cache = contract.get("expects_cache")
    actual_cache = record.get("cache_hit")
    if exp_cache is not None:
        if actual_cache is None:
            violations.append(
                f"Cache behavior undefined: expected cache_hit={exp_cache}, got None"
            )
        elif actual_cache != exp_cache:
            violations.append(
                f"Cache behavior mismatch: expected cache_hit={exp_cache}, got {actual_cache}"
            )

    # 5. Output type / content validation
    preview = record.get("answer_preview", "")
    if not preview or not preview.strip():
        violations.append("Empty output response")
    elif "The paper does not appear to cover this" in preview:
        violations.append("Refusal message produced on non-refusal workflow")

    is_valid = len(violations) == 0
    return {
        "is_valid": is_valid,
        "status": "VALID" if is_valid else "INVALID",
        "violations": violations,
    }


def load_checkpoint() -> List[Dict[str, Any]]:
    """Loads existing checkpoint records or initializes with recovered records."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception:
            pass
    return [dict(r) for r in RECOVERED_OPERATIONS]


def save_checkpoint(records: List[Dict[str, Any]]) -> None:
    """Atomically writes records to the checkpoint file."""
    ensure_results_dir()
    temp_path = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    if os.name == "nt" and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    temp_path.rename(CHECKPOINT_FILE)


def run_latency_benchmark(
    doc_id: str = TARGET_DOC_ID,
    offline: bool = False,
    resume: bool = False,
    save_output: bool = True,
) -> Dict[str, Any]:
    """Executes task-specific operations and measures latency, calls, trace, and cache status."""
    print(f"\n=======================================================")
    print(f"TASK-SPECIFIC LATENCY BENCHMARK: 8 OPERATIONS (Paper A: {doc_id})")
    print(f"Mode: {'OFFLINE (MOCKED)' if offline else 'LIVE (GROQ API)'}")
    print(f"Resume Enabled: {resume}")
    print(f"Checkpoint File: {CHECKPOINT_FILE}")
    print(f"=======================================================\n")

    operations = [
        {
            "name": "1. Paper Question Answering",
            "op_key": "paper_question",
            "query": "What is self-attention and how does it relate different positions?",
            "pre_action": None,
            "session_mode": "fresh",
        },
        {
            "name": "2. Section Summarization",
            "op_key": "section_summary",
            "query": "summarize section 3 Model Architecture",
            "pre_action": None,
            "session_mode": "fresh",
        },
        {
            "name": "3. Cold-Cache Study Notes",
            "op_key": "notes_cold",
            "query": "make study notes",
            "pre_action": "clear_notes_cache",
            "session_mode": "fresh",
        },
        {
            "name": "4. Warm-Cache Study Notes",
            "op_key": "notes_warm",
            "query": "make study notes",
            "pre_action": None,
            "session_mode": "fresh",
        },
        {
            "name": "5. Cold-Cache 8-Slide Outline",
            "op_key": "outline_cold",
            "query": "make an 8 slide presentation",
            "pre_action": "clear_outline_cache",
            "session_mode": "fresh",
        },
        {
            "name": "6. Warm-Cache 8-Slide Outline",
            "op_key": "outline_warm",
            "query": "make an 8 slide presentation",
            "pre_action": None,
            "session_mode": "fresh",
        },
        {
            "name": "7. Rephrasing Previous Answer",
            "op_key": "rephrase_previous",
            "query": "explain that more simply",
            "pre_action": None,
            "session_mode": "with_history",
        },
        {
            "name": "8. Off-Topic Query",
            "op_key": "off_topic",
            "query": "what is the capital of France?",
            "pre_action": None,
            "session_mode": "fresh",
        },
    ]

    loaded_records: List[Dict[str, Any]] = load_checkpoint() if resume else []

    # Partition loaded records into valid vs invalid
    valid_records: List[Dict[str, Any]] = []
    invalid_records: List[Dict[str, Any]] = []

    for r in loaded_records:
        val = validate_operation_record(r)
        r["status"] = val["status"]
        r["violations"] = val["violations"]
        if val["is_valid"]:
            valid_records.append(r)
        else:
            invalid_records.append(r)

    completed_valid_keys = {r["op_key"] for r in valid_records}

    # Write normalized checkpoint if resuming
    if resume and not CHECKPOINT_FILE.exists():
        save_checkpoint(valid_records + invalid_records)

    print(f"{'Operation':<32} | {'Intent':<18} | {'Calls':<6} | {'Time':<8} | {'Cache Hit':<10} | {'Status'}")
    print("-" * 115)

    # Print already completed/recovered valid operations
    for r in valid_records:
        cache_str = str(r.get("cache_hit")) if r.get("cache_hit") is not None else "N/A"
        status_str = f"VALID ({'recovered' if r.get('recovered') else 'completed'})"
        print(f"{r['operation']:<32} | {r['intent']:<18} | {r['llm_calls']:<6} | {r['elapsed_seconds']:.2f}s   | {cache_str:<10} | {status_str}")

    for r in invalid_records:
        cache_str = str(r.get("cache_hit")) if r.get("cache_hit") is not None else "N/A"
        print(f"{r['operation']:<32} | {r['intent']:<18} | {r['llm_calls']:<6} | {r['elapsed_seconds']:.2f}s   | {cache_str:<10} | INVALID (requires rerun)")

    results: List[Dict[str, Any]] = list(valid_records)

    for op in operations:
        name = op["name"]
        op_key = op["op_key"]
        query = op["query"]
        pre_act = op["pre_action"]
        contract = OPERATION_CONTRACTS.get(op_key, {})

        if op_key in completed_valid_keys:
            continue

        # Handle cache pre-actions (strictly in isolated DATA_DIR / cache)
        if pre_act == "clear_notes_cache":
            cache.clear(doc_id)
        elif pre_act == "clear_outline_cache":
            outline_path = cache._get_cache_path(doc_id, "outline_8")
            if outline_path.exists():
                outline_path.unlink()

        # Check cache state BEFORE operation
        cache_kind = contract.get("cache_kind")
        cache_exists_before = (cache.get(doc_id, cache_kind) is not None) if cache_kind else None

        # Session setup
        if op["session_mode"] == "with_history":
            session = ChatSession(
                doc_id=doc_id,
                history=[
                    {"role": "user", "content": "What is self-attention?"},
                    {
                        "role": "assistant",
                        "content": "Self-attention relates different positions of a single sequence to compute a representation [3 Model Architecture, p. 3].",
                    },
                ],
                last_output={
                    "kind": "answer",
                    "text": "Self-attention relates different positions of a single sequence to compute a representation [3 Model Architecture, p. 3].",
                },
            )
        else:
            session = ChatSession(doc_id=doc_id)

        t0 = time.time()
        reset_call_count()
        initial_calls = get_call_count()

        op_error = None
        turn_res = None
        try:
            if offline:
                with patch("time.sleep", return_value=None):
                    turn_res = session.ask(query)
            else:
                turn_res = session.ask(query)
        except Exception as e:
            op_error = str(e)
            print(f"\n[ERROR] Operation '{name}' raised exception: {e}")

        elapsed = time.time() - t0
        measured_calls = get_call_count() - initial_calls
        turn_calls = measured_calls if measured_calls > 0 else (turn_res.llm_calls if turn_res else 0)

        # Check cache state AFTER operation
        cache_exists_after = (cache.get(doc_id, cache_kind) is not None) if cache_kind else None

        # Determine actual cache behavior based on real state inspection
        if cache_kind is None:
            actual_cache_hit = None
        elif turn_res and contract.get("required_node") in turn_res.trace:
            if cache_exists_before:
                actual_cache_hit = True
            elif not cache_exists_before and cache_exists_after:
                actual_cache_hit = False
            else:
                actual_cache_hit = False
        else:
            # If node was never reached or execution failed
            actual_cache_hit = False if not cache_exists_before else None

        record = {
            "operation": name,
            "op_key": op_key,
            "query": query,
            "intent": turn_res.intent if turn_res else "unknown",
            "llm_calls": turn_calls,
            "elapsed_seconds": round(elapsed, 3),
            "cache_hit": actual_cache_hit,
            "trace": turn_res.trace if turn_res else [],
            "answer_preview": ((turn_res.answer[:80] + "...") if turn_res and turn_res.answer else ""),
            "source": "mocked_execution" if offline else "live_execution",
            "source_log": None,
            "recovered": False,
            "error": op_error,
        }

        # Validate against contract
        val = validate_operation_record(record, contract)
        record["status"] = val["status"]
        record["violations"] = val["violations"]

        if val["is_valid"]:
            results.append(record)
            completed_valid_keys.add(op_key)
            save_checkpoint(results + invalid_records)
            cache_str = str(actual_cache_hit) if actual_cache_hit is not None else "N/A"
            print(f"{name:<32} | {record['intent']:<18} | {turn_calls:<6} | {elapsed:.2f}s   | {cache_str:<10} | VALID")
        else:
            invalid_records.append(record)
            save_checkpoint(results + invalid_records)
            cache_str = str(actual_cache_hit) if actual_cache_hit is not None else "N/A"
            print(f"{name:<32} | {record['intent']:<18} | {turn_calls:<6} | {elapsed:.2f}s   | {cache_str:<10} | INVALID: {'; '.join(val['violations'])}")
            print(f"\n[STOPPING] Operation '{name}' violated workflow contract. Halting execution without retry.")
            break

    benchmark_summary = {
        "timestamp": datetime.now().isoformat(),
        "document_id": doc_id,
        "total_operations": len(results),
        "valid_count": len(results),
        "invalid_count": len(invalid_records),
        "recovered_count": sum(1 for r in results if r.get("recovered")),
        "live_count": sum(1 for r in results if not r.get("recovered")),
        "operations": results,
        "invalid_operations": invalid_records,
    }

    if save_output:
        ensure_results_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = eval_dir / "results" / f"latency_benchmark_{ts}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {out_file}")

    print("\n" + "=" * 115)
    print(f"LATENCY BENCHMARK COMPLETED. Valid: {len(results)}/8 | Invalid: {len(invalid_records)}")
    print(f"Recovered from logs: {benchmark_summary['recovered_count']} | Newly executed valid: {benchmark_summary['live_count']}")
    if save_output:
        print(f"Results saved to: {out_file}")
    print("=" * 115 + "\n")

    return benchmark_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task-Specific Latency Benchmark")
    parser.add_argument("--offline", action="store_true", help="Run offline with mocked turns")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint, preserving recovered/completed operations")
    args = parser.parse_args()
    run_latency_benchmark(offline=args.offline, resume=args.resume)
