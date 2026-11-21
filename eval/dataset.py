"""Dataset loader and validator for Phase 7b benchmark questions."""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.json"


def load_questions(
    paper_key: Optional[str] = None,
    question_id: Optional[str] = None,
    file_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Loads and filters evaluation questions.

    Args:
        paper_key: Optional filter by paper ('paper_a', 'paper_b', 'paper_c').
        question_id: Optional filter by question id (e.g., 'A1', 'B3', 'C1').
        file_path: Optional custom path to questions.json.

    Returns:
        List of question dictionaries.
    """
    target_path = file_path or QUESTIONS_FILE
    with open(target_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if paper_key:
        questions = [q for q in questions if q.get("paper_key") == paper_key]
    if question_id:
        questions = [q for q in questions if q.get("id") == question_id]

    return questions


def validate_dataset(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validates schema, uniqueness, and completeness of the evaluation questions.

    Enforces:
    - Exactly 30 questions.
    - Exactly 7 answerable (covered=True) and 3 unanswerable (covered=False) per paper.
    - Total: 21 answerable and 9 unanswerable.
    - must_contain must be a non-empty list of lists of alternative strings.

    Returns:
        Dict with validation summary: {total, paper_counts, covered_counts, valid, errors}.
    """
    required_fields = [
        "id", "paper_key", "doc_id", "paper_title", "question",
        "covered", "must_contain", "expected_answer"
    ]
    seen_ids = set()
    errors = []
    paper_counts = {}
    covered_counts = {"answerable": 0, "unanswerable": 0}
    per_paper_covered = {}

    for idx, q in enumerate(questions):
        qid = q.get("id")
        if not qid:
            errors.append(f"Question at index {idx} missing 'id'.")
            continue
        if qid in seen_ids:
            errors.append(f"Duplicate question id '{qid}' at index {idx}.")
        seen_ids.add(qid)

        for rf in required_fields:
            if rf not in q:
                errors.append(f"Question {qid} missing field '{rf}'.")

        pkey = q.get("paper_key", "unknown")
        paper_counts[pkey] = paper_counts.get(pkey, 0) + 1

        is_cov = q.get("covered", False)
        if is_cov:
            covered_counts["answerable"] += 1
        else:
            covered_counts["unanswerable"] += 1

        if pkey not in per_paper_covered:
            per_paper_covered[pkey] = {"answerable": 0, "unanswerable": 0}
        if is_cov:
            per_paper_covered[pkey]["answerable"] += 1
        else:
            per_paper_covered[pkey]["unanswerable"] += 1

        mc = q.get("must_contain")
        if is_cov:
            if not isinstance(mc, list) or len(mc) == 0:
                errors.append(f"Question {qid} must_contain must be a non-empty list.")
            else:
                for g_idx, group in enumerate(mc):
                    if not isinstance(group, list) or len(group) == 0:
                        errors.append(f"Question {qid} must_contain group {g_idx} must be a non-empty list of strings.")
        else:
            if mc is not None and not isinstance(mc, list):
                errors.append(f"Question {qid} must_contain must be a list if provided.")

    # Validate paper breakdown
    for pkey, stats in per_paper_covered.items():
        if stats["answerable"] != 7:
            errors.append(f"Paper {pkey} has {stats['answerable']} answerable questions (expected 7).")
        if stats["unanswerable"] != 3:
            errors.append(f"Paper {pkey} has {stats['unanswerable']} unanswerable questions (expected 3).")

    return {
        "total": len(questions),
        "paper_counts": paper_counts,
        "covered_counts": covered_counts,
        "per_paper_covered": per_paper_covered,
        "valid": len(errors) == 0,
        "errors": errors,
    }


if __name__ == "__main__":
    qs = load_questions()
    val = validate_dataset(qs)
    print(f"Loaded {val['total']} questions.")
    print(f"Paper counts: {val['paper_counts']}")
    print(f"Covered counts: {val['covered_counts']}")
    print(f"Per-paper breakdown: {val['per_paper_covered']}")
    if val["valid"]:
        print("Dataset validation PASSED perfectly.")
    else:
        print(f"Dataset validation FAILED with errors: {val['errors']}")
