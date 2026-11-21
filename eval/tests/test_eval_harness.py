"""Offline unit tests for Phase 7b evaluation harness.

All tests mock the LLM and run 100% offline without live API calls.
"""
import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

eval_dir = Path(__file__).resolve().parent.parent
eval_data_dir = (eval_dir / "data").resolve()
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
from eval.metrics import (
    score_factual_correctness,
    evaluate_citation_quality,
    aggregate_run_metrics,
)
from eval.storage import (
    append_raw_result,
    load_completed_question_ids,
    load_raw_results,
)
from eval.report import generate_summary_json, generate_markdown_report
from eval.baseline_rag import run_baseline_rag
from eval.agentic_runner import run_agentic_turn


class TestDatasetValidation:
    def test_full_dataset_structure(self):
        questions = load_questions()
        val = validate_dataset(questions)
        assert val["valid"] is True, f"Dataset errors: {val['errors']}"
        assert val["total"] == 30, f"Expected 30 questions, got {val['total']}"
        assert val["paper_counts"] == {"paper_a": 10, "paper_b": 10, "paper_c": 10}
        assert val["covered_counts"] == {"answerable": 21, "unanswerable": 9}

    def test_unique_ids_and_paper_prefixes(self):
        questions = load_questions()
        ids = [q["id"] for q in questions]
        assert len(ids) == len(set(ids)), "Duplicate question IDs detected"

        expected_ids = [f"A{i}" for i in range(1, 11)] + [f"B{i}" for i in range(1, 11)] + [f"C{i}" for i in range(1, 11)]
        assert sorted(ids) == sorted(expected_ids)

    def test_discrepancy_questions_present(self):
        questions = {q["id"]: q for q in load_questions()}
        assert "A2" in questions
        assert "B3" in questions
        assert "C1" in questions

        # Verify A2 must_contain has both 41.0 and 41.8 groups
        a2_groups = questions["A2"]["must_contain"]
        assert any("41.0" in g for g in a2_groups)
        assert any("41.8" in g for g in a2_groups)

        # Verify B3 must_contain has 54,000 and 70
        b3_groups = questions["B3"]["must_contain"]
        assert any("54,000" in g or "54000" in g for g in b3_groups)
        assert any("70" in g for g in b3_groups)

        # Verify C1 must_contain has both 4.1 and 4.3 groups
        c1_groups = questions["C1"]["must_contain"]
        assert any("4.1" in g for g in c1_groups)
        assert any("4.3" in g for g in c1_groups)

    def test_refusal_questions_present(self):
        questions = {q["id"]: q for q in load_questions()}
        assert questions["A8"]["covered"] is False
        assert questions["A9"]["covered"] is False
        assert questions["A10"]["covered"] is False
        assert questions["B8"]["covered"] is False
        assert questions["B9"]["covered"] is False
        assert questions["B10"]["covered"] is False
        assert questions["C8"]["covered"] is False
        assert questions["C9"]["covered"] is False
        assert questions["C10"]["covered"] is False


class TestMetricsAndScoring:
    def test_factual_correctness_exact_match(self):
        answer = "The Transformer big model achieved 28.4 BLEU on English-to-German [6 Results, p. 8]."
        gold = "The Transformer achieved 28.4 BLEU [6 Results, p. 8]."
        res = score_factual_correctness(answer, gold, [["28.4"], ["BLEU"]], covered=True)
        assert res["score"] == 1.0
        assert res["is_correct"] is True
        assert res["matched_groups"] == [0, 1]
        assert res["missing_groups"] == []

    def test_factual_correctness_partial_and_fail(self):
        # 1 of 3 groups matched -> 33% ratio -> is_correct False
        answer = "The model has 1.9 million parameters."
        res = score_factual_correctness(answer, "Gold answer", [["1.9"], ["0.32"], ["38"]], covered=True)
        assert res["score"] == pytest.approx(1 / 3)
        assert res["is_correct"] is False
        assert res["matched_groups"] == [0]
        assert res["missing_groups"] == [1, 2]

    def test_refusal_scoring_true_refusal(self):
        answer = "The paper does not appear to cover this topic."
        res = score_factual_correctness(answer, "Gold refusal", [], covered=False)
        assert res["score"] == 1.0
        assert res["is_correct"] is True
        assert res["is_refusal"] is True
        assert res["correct_refusal"] is True

    def test_refusal_scoring_false_answer_on_unanswerable(self):
        answer = "The model achieved 85.4% top-1 accuracy on ImageNet."
        res = score_factual_correctness(answer, "Gold refusal", [], covered=False)
        assert res["score"] == 0.0
        assert res["is_correct"] is False
        assert res["is_refusal"] is False
        assert res["correct_refusal"] is False

    def test_refusal_scoring_false_refusal_on_answerable(self):
        answer = "The paper does not appear to cover this."
        res = score_factual_correctness(answer, "Gold answer", [["28.4"], ["BLEU"]], covered=True)
        assert res["score"] == 0.0
        assert res["is_correct"] is False
        assert res["is_refusal"] is True
        assert res["false_refusal"] is True

    def test_citation_evaluation(self):
        check = {
            "cited": ["[6 Results, p. 8]", "[3 Model, p. 2]"],
            "supported": ["[6 Results, p. 8]"],
            "unsupported": ["[3 Model, p. 2]"],
        }
        eval_c = evaluate_citation_quality(check)
        assert eval_c["num_cited"] == 2
        assert eval_c["num_supported"] == 1
        assert eval_c["num_unsupported"] == 1
        assert eval_c["citation_precision"] == 0.5
        assert eval_c["fully_supported"] is False

    def test_aggregate_run_metrics(self):
        sample_results = [
            {
                "question_id": "A1",
                "paper_key": "paper_a",
                "is_correct": True,
                "score": 1.0,
                "covered": True,
                "refused": False,
                "has_citations": True,
                "num_cited": 1,
                "num_supported": 1,
                "seconds": 1.5,
                "llm_calls": 1,
            },
            {
                "question_id": "A10",
                "paper_key": "paper_a",
                "is_correct": True,
                "score": 1.0,
                "covered": False,
                "refused": True,
                "correct_refusal": True,
                "has_citations": False,
                "num_cited": 0,
                "num_supported": 0,
                "seconds": 0.5,
                "llm_calls": 1,
            },
            {
                "question_id": "B1",
                "paper_key": "paper_b",
                "is_correct": False,
                "score": 0.0,
                "covered": True,
                "refused": False,
                "has_citations": True,
                "num_cited": 1,
                "num_supported": 0,
                "seconds": 2.0,
                "llm_calls": 2,
            },
        ]
        agg = aggregate_run_metrics(sample_results)
        assert agg["total_questions"] == 3
        assert agg["correct_answers"] == 2
        assert pytest.approx(agg["accuracy"]) == 2 / 3
        assert agg["refusal_metrics"]["precision"] == 1.0
        assert agg["refusal_metrics"]["recall"] == 1.0
        assert agg["latency_metrics"]["avg_seconds"] > 0


class TestStorageAndRecovery:
    def test_incremental_append_and_recovery(self, tmp_path):
        run_file = tmp_path / "raw_test_run.json"

        # Record 1
        r1 = {"question_id": "A1", "score": 1.0, "is_correct": True}
        append_raw_result(run_file, r1)

        completed = load_completed_question_ids(run_file)
        assert completed == {"A1"}

        # Record 2
        r2 = {"question_id": "A2", "score": 1.0, "is_correct": True}
        append_raw_result(run_file, r2)

        completed = load_completed_question_ids(run_file)
        assert completed == {"A1", "A2"}

        all_records = load_raw_results(run_file)
        assert len(all_records) == 2


class TestOfflineRunners:
    @patch("eval.baseline_rag.search")
    @patch("eval.baseline_rag.generate")
    def test_baseline_rag_answer_flow(self, mock_gen, mock_search):
        mock_search.return_value = [
            {"text": "The Transformer big model achieved 28.4 BLEU.", "section": "6 Results", "page_start": 8, "page_end": 8, "score": 0.85}
        ]
        mock_gen.return_value = "The Transformer (big) achieved 28.4 BLEU [6 Results, p. 8]."

        res = run_baseline_rag(doc_id="bdfaa68d8984", question="What BLEU score?")
        assert res["refused"] is False
        assert "28.4" in res["answer"]
        assert len(res["chunks"]) == 1
        assert res["llm_calls"] == 1
        assert res["seconds"] >= 0

    @patch("eval.baseline_rag.search")
    @patch("eval.baseline_rag.generate")
    def test_baseline_rag_refusal_prompt_grounded(self, mock_gen, mock_search):
        # Even with irrelevant chunks, baseline calls LLM once and relies on prompt refusal
        mock_search.return_value = [
            {"text": "Irrelevant text", "section": "Unknown", "page_start": 1, "page_end": 1, "score": 0.10}
        ]
        mock_gen.return_value = "The paper does not appear to cover this."

        res = run_baseline_rag(doc_id="bdfaa68d8984", question="ImageNet classification?")
        assert res["refused"] is True
        assert res["llm_calls"] == 1
        assert "does not appear to cover" in res["answer"]

    @patch("app.agents.graph.compiled_app.invoke")
    def test_agentic_runner_flow(self, mock_invoke):
        mock_decision = MagicMock(intent="paper_question", standalone_query="BLEU score")
        mock_invoke.return_value = {
            "result_text": "The Transformer reached 28.4 BLEU [6 Results, p. 8].",
            "decision": mock_decision,
            "trace": ["router_node", "qa_retrieve", "qa_answer", "update_memory"],
            "history": [],
            "last_output": {"kind": "answer", "text": "sample"},
            "qa_result": {
                "answer": "The Transformer reached 28.4 BLEU [6 Results, p. 8].",
                "raw_answer": "The Transformer reached 28.4 BLEU [6 Results, p. 8].",
                "citation_check": {"cited": ["[6 Results, p. 8]"], "supported": ["[6 Results, p. 8]"], "unsupported": []},
                "refused": False,
            }
        }

        res = run_agentic_turn(doc_id="bdfaa68d8984", question="What BLEU score?")

        assert res["intent"] == "paper_question"
        assert "28.4" in res["answer"]
        assert "router_node" in res["trace"]
        assert res["refused"] is False


class TestReportGeneration:
    def test_report_generation_from_results(self, tmp_path):
        sample_results = [
            {
                "question_id": "A1",
                "paper_key": "paper_a",
                "is_correct": True,
                "score": 1.0,
                "covered": True,
                "refused": False,
                "has_citations": True,
                "num_cited": 1,
                "num_supported": 1,
                "seconds": 1.2,
                "llm_calls": 1,
            }
        ]
        json_path = tmp_path / "summary.json"
        md_path = tmp_path / "report.md"

        gen_json = generate_summary_json(baseline_results=sample_results, agentic_results=sample_results, output_path=json_path)
        gen_md = generate_markdown_report(baseline_results=sample_results, agentic_results=sample_results, output_path=md_path)

        assert gen_json.exists()
        assert gen_md.exists()

        with open(gen_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "baseline" in data
            assert "agentic" in data

        with open(gen_md, "r", encoding="utf-8") as f:
            content = f.read()
            assert "# Phase 7b: System Evaluation & Benchmarking Report" in content
            assert "Experimental Configuration" in content
            assert "Benchmark Comparison Summary" in content
