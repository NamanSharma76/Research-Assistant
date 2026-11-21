"""Incremental result persistence and run recovery for Phase 7b benchmark.

Recovery Rules:
- First successful raw answers cannot be overwritten by subsequent runs.
- Resume retries only missing or infrastructure-failed cases.
- Atomic file writes guarantee that interrupted runs lose at most the currently in-flight query.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def ensure_results_dir() -> Path:
    """Ensures results directory exists."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def get_run_filepath(system_name: str, timestamp: Optional[str] = None) -> Path:
    """Generates unique file path for a run."""
    ensure_results_dir()
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"raw_{system_name}_{ts}.json"


def load_raw_results(file_path: Path) -> List[Dict[str, Any]]:
    """Loads existing raw results from file."""
    if not file_path.exists():
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def load_completed_question_ids(file_path: Path) -> Set[str]:
    """Returns set of question IDs that completed successfully without infrastructure error.

    Missing or infrastructure-failed queries are excluded so that --resume will re-try them.
    """
    records = load_raw_results(file_path)
    return {
        r.get("question_id")
        for r in records
        if r.get("question_id") and not r.get("error")
    }


def append_raw_result(file_path: Path, record: Dict[str, Any]) -> None:
    """Appends or updates a question record in the raw results file atomically.

    Guarantees:
    - First successful raw answers cannot be overwritten by subsequent attempts.
    - Infrastructure-failed records can be updated upon retry.
    """
    records = load_raw_results(file_path)
    qid = record.get("question_id")

    existing_idx = next((i for i, r in enumerate(records) if r.get("question_id") == qid), None)
    if existing_idx is not None:
        existing_record = records[existing_idx]
        # Protect successful answers: only overwrite if previous record had an error
        if existing_record.get("error"):
            records[existing_idx] = record
    else:
        records.append(record)

    temp_path = file_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    if os.name == "nt" and file_path.exists():
        file_path.unlink()
    temp_path.rename(file_path)


def find_latest_run_file(system_name: str) -> Optional[Path]:
    """Finds the most recent raw results file for a given system to resume."""
    ensure_results_dir()
    candidates = list(RESULTS_DIR.glob(f"raw_{system_name}_*.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
