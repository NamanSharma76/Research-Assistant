"""Evaluation metrics, scoring engine, and normalization for Phase 7b benchmarking.

Scoring & Verification Rules:
- All required must_contain groups must match for strict correctness.
- Alternatives within a group use OR matching.
- Matching normalizes Unicode dashes (-, \u2013, \u2014), non-breaking spaces (\u00a0), and case.
- Correct refusals and false refusals are measured separately.
- Infrastructure failures are never counted as semantic incorrect answers.
- Citation reference validity (format + document chunk match) is distinguished from semantic factual entailment.
- Metric denominators explicitly report question counts and subset sizes.
"""
import re
import unicodedata
from typing import Dict, Any, List, Optional
from app.rag.prompts import is_refusal


def normalize_text(text: str) -> str:
    """Normalizes text by unifiying Unicode dashes, spaces, and casing.

    Replaces en-dashes, em-dashes, and minus signs with standard hyphens;
    replaces non-breaking and thin spaces with standard spaces; strips excess whitespace.
    """
    if not text:
        return ""
    # Normalize unicode forms
    text = unicodedata.normalize("NFKC", text)
    # Replace unicode dashes
    text = re.sub(r"[\u2010\u2012\u2013\u2014\u2015\u2212]", "-", text)
    # Replace non-breaking and special spaces
    text = re.sub(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]", " ", text)
    # Lowercase and collapse consecutive spaces
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def score_factual_correctness(
    answer: str,
    gold_answer: str,
    must_contain: List[List[str]],
    covered: bool,
) -> Dict[str, Any]:
    """Evaluates answer correctness against gold reference and must_contain groups.

    Args:
        answer: Generated answer string.
        gold_answer: Reference gold answer.
        must_contain: List of groups of alternative strings.
                      Matching any alternative satisfies the group.
                      All groups must be satisfied for strict correctness.
        covered: Boolean flag (True if question is answerable from paper, False if unanswerable).

    Returns:
        Dict: {
            "score": float (0.0 to 1.0),
            "is_correct": bool,
            "matched_groups": List[int],
            "missing_groups": List[int],
            "is_refusal": bool,
            "correct_refusal": Optional[bool],
            "false_refusal": bool,
        }
    """
    if not answer or not answer.strip():
        return {
            "score": 0.0,
            "is_correct": False,
            "matched_groups": [],
            "missing_groups": list(range(len(must_contain))),
            "is_refusal": False,
            "correct_refusal": False if not covered else None,
            "false_refusal": False,
        }

    ans_norm = normalize_text(answer)
    ans_refusal = is_refusal(answer)

    # Case 1: Unanswerable question (covered=False) -> expects refusal
    if not covered:
        if ans_refusal:
            return {
                "score": 1.0,
                "is_correct": True,
                "matched_groups": [0],
                "missing_groups": [],
                "is_refusal": True,
                "correct_refusal": True,
                "false_refusal": False,
            }
        else:
            return {
                "score": 0.0,
                "is_correct": False,
                "matched_groups": [],
                "missing_groups": [0],
                "is_refusal": False,
                "correct_refusal": False,
                "false_refusal": False,
            }

    # Case 2: Answerable question (covered=True), but system gave a refusal (False Refusal)
    if ans_refusal:
        return {
            "score": 0.0,
            "is_correct": False,
            "matched_groups": [],
            "missing_groups": list(range(len(must_contain))),
            "is_refusal": True,
            "correct_refusal": False,
            "false_refusal": True,
        }

    # Case 3: Answerable question with an answer -> evaluate must_contain groups
    matched_groups = []
    missing_groups = []

    for g_idx, group in enumerate(must_contain):
        group_matched = False
        for alt in group:
            alt_norm = normalize_text(alt)
            # Match as whole word/phrase or substring if punctuation-heavy
            pattern = r"(?:\b|\W|^)" + re.escape(alt_norm) + r"(?:\b|\W|$)"
            if re.search(pattern, ans_norm) or alt_norm in ans_norm:
                group_matched = True
                break
        if group_matched:
            matched_groups.append(g_idx)
        else:
            missing_groups.append(g_idx)

    total_groups = len(must_contain)
    ratio = len(matched_groups) / total_groups if total_groups > 0 else 1.0
    is_correct = (len(matched_groups) == total_groups)

    return {
        "score": ratio,
        "is_correct": is_correct,
        "matched_groups": matched_groups,
        "missing_groups": missing_groups,
        "is_refusal": False,
        "correct_refusal": None,
        "false_refusal": False,
    }


def evaluate_citation_quality(citation_check: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates citation presence, grounding validity, and unsupported citations."""
    cited = citation_check.get("cited", [])
    supported = citation_check.get("supported", [])
    unsupported = citation_check.get("unsupported", [])

    num_cited = len(cited)
    num_supported = len(supported)
    num_unsupported = len(unsupported)

    precision = (num_supported / num_cited) if num_cited > 0 else 1.0

    return {
        "num_cited": num_cited,
        "num_supported": num_supported,
        "num_unsupported": num_unsupported,
        "citation_precision": precision,
        "has_citations": num_cited > 0,
        "fully_supported": (num_cited > 0 and num_unsupported == 0),
    }


def aggregate_run_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes comprehensive benchmark summary metrics with explicit denominators."""
    total = len(results)
    if total == 0:
        return {"total": 0}

    infra_failures = [r for r in results if r.get("error")]
    valid_runs = [r for r in results if not r.get("error")]
    valid_count = len(valid_runs)

    if valid_count == 0:
        return {
            "total_questions": total,
            "valid_evaluations": 0,
            "infrastructure_failures": len(infra_failures),
            "correct_answers": 0,
            "accuracy": 0.0,
            "overall_accuracy": 0.0,
            "mean_score": 0.0,
            "denominators": {"total": total, "valid": 0, "answerable": 0, "unanswerable": 0},
            "answerable_metrics": {"count": 0, "correct": 0, "accuracy": 0.0, "false_refusals": 0},
            "refusal_metrics": {"expected_count": 0, "triggered_count": 0, "true_refusals": 0, "precision": 0.0, "recall": 0.0},
            "citation_metrics": {"total_cited": 0, "total_supported": 0, "precision": 0.0, "answers_with_citations": 0, "citation_rate": 0.0},
            "latency_metrics": {"avg_seconds": 0.0, "p50_seconds": 0.0, "p95_seconds": 0.0},
            "llm_call_metrics": {"total_calls": 0, "avg_calls_per_turn": 0.0},
            "per_paper": {},
        }

    correct_count = sum(1 for r in valid_runs if r.get("is_correct", False))
    scores = [r.get("score", 0.0) for r in valid_runs]
    mean_score = sum(scores) / valid_count

    # Split Answerable (covered=True) vs Unanswerable (covered=False)
    answerable_runs = [r for r in valid_runs if r.get("covered", True)]
    unanswerable_runs = [r for r in valid_runs if not r.get("covered", True)]

    # Answerable stats
    ans_count = len(answerable_runs)
    ans_correct = sum(1 for r in answerable_runs if r.get("is_correct", False))
    ans_accuracy = (ans_correct / ans_count) if ans_count > 0 else 0.0
    false_refusals = sum(1 for r in answerable_runs if r.get("false_refusal", False))

    # Refusal stats
    unans_count = len(unanswerable_runs)
    true_refusals = sum(1 for r in unanswerable_runs if r.get("correct_refusal", False))
    refusal_recall = (true_refusals / unans_count) if unans_count > 0 else 1.0

    all_refusals_triggered = sum(1 for r in valid_runs if r.get("refused", False))
    refusal_precision = (true_refusals / all_refusals_triggered) if all_refusals_triggered > 0 else 1.0

    # Citation metrics
    citations_present = sum(1 for r in valid_runs if r.get("has_citations", False))
    supported_citations = sum(r.get("num_supported", 0) for r in valid_runs)
    total_citations = sum(r.get("num_cited", 0) for r in valid_runs)
    citation_precision = (supported_citations / total_citations) if total_citations > 0 else 1.0

    # Latency and call metrics
    latencies = sorted([r.get("seconds", 0.0) for r in valid_runs])
    avg_latency = sum(latencies) / valid_count
    p50_latency = latencies[valid_count // 2]
    p95_latency = latencies[int(valid_count * 0.95)] if valid_count > 1 else latencies[0]

    llm_calls = [r.get("llm_calls", 0) for r in valid_runs]
    total_calls = sum(llm_calls)
    avg_calls = total_calls / valid_count

    # Per-paper accuracy breakdown
    per_paper = {}
    for r in valid_runs:
        pkey = r.get("paper_key", "unknown")
        if pkey not in per_paper:
            per_paper[pkey] = {"total": 0, "correct": 0, "answerable_correct": 0, "answerable_total": 0, "refusal_correct": 0, "refusal_total": 0}
        per_paper[pkey]["total"] += 1
        if r.get("is_correct", False):
            per_paper[pkey]["correct"] += 1
        if r.get("covered", True):
            per_paper[pkey]["answerable_total"] += 1
            if r.get("is_correct", False):
                per_paper[pkey]["answerable_correct"] += 1
        else:
            per_paper[pkey]["refusal_total"] += 1
            if r.get("is_correct", False):
                per_paper[pkey]["refusal_correct"] += 1

    for pkey, stats in per_paper.items():
        stats["overall_accuracy"] = (stats["correct"] / stats["total"]) if stats["total"] > 0 else 0.0
        stats["answerable_accuracy"] = (stats["answerable_correct"] / stats["answerable_total"]) if stats["answerable_total"] > 0 else 0.0
        stats["refusal_accuracy"] = (stats["refusal_correct"] / stats["refusal_total"]) if stats["refusal_total"] > 0 else 0.0

    return {
        "total_questions": total,
        "valid_evaluations": valid_count,
        "infrastructure_failures": len(infra_failures),
        "correct_answers": correct_count,
        "accuracy": correct_count / valid_count,
        "overall_accuracy": correct_count / valid_count,
        "mean_score": mean_score,
        "denominators": {
            "total_questions": total,
            "valid_questions": valid_count,
            "answerable_questions": ans_count,
            "unanswerable_questions": unans_count,
        },
        "answerable_metrics": {
            "total_answerable": ans_count,
            "correct_answerable": ans_correct,
            "accuracy": ans_accuracy,
            "false_refusals": false_refusals,
        },
        "refusal_metrics": {
            "expected_unanswerable": unans_count,
            "triggered_refusals": all_refusals_triggered,
            "true_refusals": true_refusals,
            "precision": refusal_precision,
            "recall": refusal_recall,
        },
        "citation_metrics": {
            "total_cited": total_citations,
            "total_supported": supported_citations,
            "precision": citation_precision,
            "answers_with_citations": citations_present,
            "citation_rate": citations_present / valid_count,
        },
        "latency_metrics": {
            "avg_seconds": round(avg_latency, 2),
            "p50_seconds": round(p50_latency, 2),
            "p95_seconds": round(p95_latency, 2),
        },
        "llm_call_metrics": {
            "total_calls": total_calls,
            "avg_calls_per_turn": round(avg_calls, 2),
        },
        "per_paper": per_paper,
    }
