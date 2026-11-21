"""CLI runner for Phase 7b benchmark execution."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from unittest.mock import patch, MagicMock

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

from eval.dataset import load_questions, validate_dataset
from eval.baseline_rag import run_baseline_rag
from eval.agentic_runner import run_agentic_turn
from eval.metrics import score_factual_correctness, evaluate_citation_quality
from eval.storage import (
    get_run_filepath,
    append_raw_result,
    load_completed_question_ids,
    load_raw_results,
    find_latest_run_file,
)
from eval.report import generate_summary_json, generate_markdown_report


def mock_generate_fn(prompt: str, system: str = None, model: str = None, **kwargs) -> str:
    """Offline deterministic mock generator for test execution without Groq API calls."""
    p_lower = (prompt or "").lower()
    # Query rewriter response
    if "research paper query optimizer" in p_lower:
        return "dilated temporal convolutions electricity forecasting"

    # Refusals for unanswerable / out-of-scope questions
    if any(k in p_lower for k in [
        "imagenet", "solar", "bert", "gpt-3", "tpu", "iphone",
        "satellite", "wind", "gpu acceleration", "bleu score of leafscan", "bleu score of gridtcn"
    ]):
        return "The paper does not appear to cover this."

    # Paper A
    if "english-to-german" in p_lower:
        return "The Transformer (big) achieved a BLEU score of 28.4 on English-to-German [6 Results, p. 8]."
    if "english-to-french" in p_lower or "french" in p_lower:
        return "Section 6.1 text reports 41.0 BLEU while Table 2 reports 41.8 BLEU [6 Results, p. 8]."
    if "d_model" in p_lower:
        return "In the base Transformer, d_model is 512 [3 Model Architecture, p. 3]."
    if "attention heads" in p_lower:
        return "The model uses 8 parallel attention heads [3 Model Architecture, p. 4]."
    if "optimizer and warmup" in p_lower or "warmup steps" in p_lower:
        return "The Adam optimizer was used with 4000 warmup steps [5 Training, p. 7]."
    if "gpus were used" in p_lower or ("gpus" in p_lower and "training" in p_lower):
        return "Training was performed on 8 NVIDIA P100 GPUs [5 Training, p. 7]."
    if "layers does the encoder" in p_lower or "encoder have" in p_lower:
        return "The encoder consists of 6 layers [3 Model Architecture, p. 2]."

    # Paper B
    if "accuracy does leafscan get on the test" in p_lower:
        return "LeafScan reached 97.8% test accuracy on the PlantLeaf-38 test set [5 Results, p. 2]."
    if "learning rate and batch size were used" in p_lower:
        return "Trained with learning rate 1e-3 and batch size 64 [4 Experimental Setup, p. 2]."
    if "how many images are in the dataset and how are they split" in p_lower or "54,000" in p_lower:
        return "The dataset contains 54,000 images split 70/15/15 into train, val, and test sets [4 Experimental Setup, p. 2]."
    if "field photographs" in p_lower:
        return "On field photographs, accuracy was 81.3% [5 Results, p. 3]."
    if "parameters does leafscan have" in p_lower:
        return "LeafScan has 1.9 million parameters [Abstract, p. 1]."
    if "baseline models were compared" in p_lower:
        return "The baselines compared were MobileNetV2, EfficientNet, and ResNet [5 Results, p. 2]."
    if "cutmix is removed" in p_lower:
        return "When CutMix was removed in the ablation study, accuracy dropped to 97.1% [5 Results, p. 3]."

    # Paper C
    if "mape does gridtcn get" in p_lower:
        return "Section 5.1 states 4.1% MAPE while Table 2 reports 4.3% MAPE [5 Results, p. 2, Table 2]."
    if "parameters does gridtcn have" in p_lower:
        return "GridTCN has 0.8 million parameters [Abstract, p. 1]."
    if "optimiser, learning rate and batch size" in p_lower:
        return "Trained with the Adam optimiser, a learning rate of 2e-3, and batch size 128 [3 Our Approach, p. 2]."
    if "how many substations" in p_lower:
        return "The dataset contains 120 substations covering 2021 to 2023 [4 Data, p. 2]."
    if "lstm reach when 20%" in p_lower or ("lstm" in p_lower and "missing" in p_lower):
        return "When 20% of readings are missing, the LSTM reaches a MAPE of 7.3% [5 Results, p. 3]."
    if "how long did training take and on what hardware" in p_lower or ("training take" in p_lower and "hardware" in p_lower):
        return "Training took about 3.5 hours on an 8-core CPU [3 Our Approach, p. 2]."
    if "ablation study" in p_lower and "matters most" in p_lower:
        return "Removing dilated convolutions matters most, degrading MAPE to 5.2% [5 Results, p. 3]."

    return "The model reports strong performance across benchmarks [Abstract, p. 1]."


def mock_generate_json_fn(prompt: str, system: str = None, model: str = None, schema=None, **kwargs) -> dict:
    """Offline deterministic mock JSON generator for Router."""
    from app.agents.schemas import RouterDecision
    p_lower = (prompt or "").lower()
    intent = "paper_question"
    if "summarize" in p_lower:
        intent = "summarize"
    elif "notes" in p_lower:
        intent = "notes"
    elif "outline" in p_lower:
        intent = "outline"
    return RouterDecision(
        intent=intent,
        standalone_query=prompt[:100],
        confidence=0.95,
        reason="Mocked router classification",
    ).model_dump()


class MockChatCompletions:
    def create(self, messages, model=None, response_format=None, **kwargs):
        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")
        if response_format and response_format.get("type") == "json_object":
            content = json.dumps(mock_generate_json_fn(user_msg))
        else:
            content = mock_generate_fn(user_msg)

        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_res = MagicMock()
        mock_res.choices = [mock_choice]
        return mock_res


class MockGroqClient:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MockChatCompletions()
        self.api_key = "mock_key"


def run_system_evaluation(
    system_name: str,
    questions: list,
    offline: bool = False,
    resume_file: Optional[Path] = None,
) -> list:
    """Runs benchmark evaluation for either 'baseline' or 'agentic' system."""
    run_file = resume_file or get_run_filepath(system_name)
    completed_ids = load_completed_question_ids(run_file) if resume_file else set()

    print(f"\n=======================================================")
    print(f"RUNNING BENCHMARK: {system_name.upper()} SYSTEM")
    print(f"Mode: {'OFFLINE (MOCKED)' if offline else 'LIVE (GROQ API)'}")
    print(f"Target Questions: {len(questions)} (Already completed: {len(completed_ids)})")
    print(f"Results File: {run_file}")
    print(f"=======================================================\n")

    results = load_raw_results(run_file) if resume_file else []
    mock_client_instance = MockGroqClient()

    for idx, q in enumerate(questions, 1):
        qid = q["id"]
        if qid in completed_ids:
            print(f"[{idx}/{len(questions)}] Skipping completed question {qid}")
            continue

        doc_id = q["doc_id"]
        q_text = q["question"]
        gold_ans = q["expected_answer"]
        must_contain = q["must_contain"]
        covered = q["covered"]

        print(f"[{idx}/{len(questions)}] Processing {qid} ({q['paper_title']}): {q_text[:60]}...")

        try:
            if offline:
                with patch("app.llm.get_groq_client", return_value=mock_client_instance), \
                     patch("time.sleep", return_value=None):
                    if system_name == "baseline":
                        run_res = run_baseline_rag(doc_id=doc_id, question=q_text)
                    else:
                        run_res = run_agentic_turn(doc_id=doc_id, question=q_text)
            else:
                if system_name == "baseline":
                    run_res = run_baseline_rag(doc_id=doc_id, question=q_text)
                else:
                    run_res = run_agentic_turn(doc_id=doc_id, question=q_text)

            score_res = score_factual_correctness(
                answer=run_res["answer"],
                gold_answer=gold_ans,
                must_contain=must_contain,
                covered=covered,
            )
            cite_res = evaluate_citation_quality(run_res.get("citation_check", {}))

            record = {
                "question_id": qid,
                "paper_key": q["paper_key"],
                "doc_id": doc_id,
                "question": q_text,
                "expected_answer": gold_ans,
                "must_contain": must_contain,
                "covered": covered,
                "answer": run_res["answer"],
                "raw_answer": run_res.get("raw_answer", run_res["answer"]),
                "refused": run_res["refused"],
                "score": score_res["score"],
                "is_correct": score_res["is_correct"],
                "matched_groups": score_res["matched_groups"],
                "missing_groups": score_res["missing_groups"],
                "correct_refusal": score_res.get("correct_refusal"),
                "false_refusal": score_res.get("false_refusal", False),
                "llm_calls": run_res["llm_calls"],
                "seconds": round(run_res["seconds"], 2),
                "num_cited": cite_res["num_cited"],
                "num_supported": cite_res["num_supported"],
                "has_citations": cite_res["has_citations"],
                "error": None,
            }
            if "trace" in run_res:
                record["trace"] = run_res["trace"]
            if "standalone_query" in run_res:
                record["standalone_query"] = run_res["standalone_query"]
            if "pre_citations" in run_res:
                record["pre_citations"] = run_res["pre_citations"]
            if "post_citations" in run_res:
                record["post_citations"] = run_res["post_citations"]

            status_str = "CORRECT" if record["is_correct"] else "MISSED"
            print(f"     -> {status_str} (score: {record['score']:.2f}, calls: {record['llm_calls']}, time: {record['seconds']}s)")

        except Exception as e:
            print(f"     -> INFRASTRUCTURE ERROR: {str(e)}")
            record = {
                "question_id": qid,
                "paper_key": q["paper_key"],
                "doc_id": doc_id,
                "question": q_text,
                "expected_answer": gold_ans,
                "must_contain": must_contain,
                "covered": covered,
                "answer": "",
                "raw_answer": "",
                "refused": False,
                "score": 0.0,
                "is_correct": False,
                "llm_calls": 0,
                "seconds": 0.0,
                "error": str(e),
            }

        append_raw_result(run_file, record)
        results.append(record)

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 7b Evaluation and Benchmarking CLI")
    parser.add_argument("--system", choices=["baseline", "agentic", "both"], default="both", help="System to evaluate")
    parser.add_argument("--paper", choices=["all", "paper_a", "paper_b", "paper_c"], default="all", help="Paper filter")
    parser.add_argument("--question-id", default=None, help="Filter to specific question id (e.g. A2, B3, C1)")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode with mocked LLM (0 Groq calls)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest incomplete run")
    args = parser.parse_args()

    paper_filter = None if args.paper == "all" else args.paper
    questions = load_questions(paper_key=paper_filter, question_id=args.question_id)

    val = validate_dataset(questions)
    if not val["valid"]:
        print(f"Dataset validation failed: {val['errors']}")
        sys.exit(1)

    print(f"Dataset loaded: {len(questions)} questions across papers: {val['paper_counts']}")

    baseline_results = None
    agentic_results = None

    if args.system in ["baseline", "both"]:
        resume_f = find_latest_run_file("baseline") if args.resume else None
        baseline_results = run_system_evaluation("baseline", questions, offline=args.offline, resume_file=resume_f)

    if args.system in ["agentic", "both"]:
        resume_f = find_latest_run_file("agentic") if args.resume else None
        agentic_results = run_system_evaluation("agentic", questions, offline=args.offline, resume_file=resume_f)

    summary_file = generate_summary_json(baseline_results, agentic_results)
    report_file = generate_markdown_report(baseline_results, agentic_results)

    print("\n=======================================================")
    print("EVALUATION RUN COMPLETE")
    print(f"Summary JSON: {summary_file}")
    print(f"Markdown Report: {report_file}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
