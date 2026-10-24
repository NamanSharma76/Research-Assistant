"""CLI tool for calibrating retrieval threshold on an academic research paper."""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.rag.retrieval import search
from app.config import RELEVANCE_THRESHOLD

# 15 questions the paper DOES answer (written for 'Attention Is All You Need')
COVERED_QUESTIONS = [
    "What BLEU score did the Transformer big model achieve on English-to-German translation?",
    "How many layers are in the encoder and decoder of the base Transformer?",
    "How many attention heads are used in the multi-head attention mechanism?",
    "What optimizer and learning rate schedule are used during training?",
    "How are positional encodings computed using sine and cosine functions?",
    "What dataset was used for the English-to-German translation experiments?",
    "What is the dimensionality d_model of the input and output embeddings?",
    "Why does the Transformer eliminate recurrence and convolution?",
    "What dropout rate is applied to the residual connections and embeddings?",
    "What is the formula for scaled dot-product attention?",
    "What is the maximum path length for self-attention layers?",
    "What label smoothing value is used during training?",
    "How many GPUs were used to train the base model and for how many hours?",
    "What is the feed-forward network inner dimension d_ff?",
    "What is the key advantage of Multi-Head Attention over a single attention function?",
]

# 15 questions that are NOT covered (harder nearby negatives + unrelated knowledge)
NOT_COVERED_QUESTIONS = [
    "What accuracy does BERT get on GLUE?",
    "How many parameters does GPT-3 have?",
    "How does ResNet use skip connections?",
    "What is the capital of France?",
    "What is photosynthesis in plant cells?",
    "What is the batch size used in training the AlphaFold model?",
    "How does YOLO perform real-time bounding box prediction in images?",
    "What was the reinforcement learning algorithm used in AlphaGo?",
    "What is the chemical formula of glucose?",
    "Who is the current prime minister of the United Kingdom?",
    "What learning rate does BERT use during pre-training?",
    "How many layers does GPT-2 have?",
    "What is the RoBERTa training batch size?",
    "What perplexity does LLaMA achieve on WikiText?",
    "How does diffusion generate images in DALL-E 2?",
]


def calibrate(doc_id: str, threshold: float = RELEVANCE_THRESHOLD) -> None:
    """Evaluates top-1 retrieval scores for 15 covered and 15 non-covered questions."""
    print(f"=== Calibrating Retrieval Relevance Threshold for doc_id: {doc_id} ===")
    print(f"Current Configured Pre-Filter Threshold: {threshold:.4f}\n")

    # Table Header
    header = f"{'#':<3} | {'Type':<12} | {'Question':<55} | {'Top-1 Score':<11} | {'Pass Threshold?':<15}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    covered_scores: List[float] = []
    not_covered_scores: List[float] = []

    # Run covered questions
    idx = 1
    covered_filtered_count = 0
    for q in COVERED_QUESTIONS:
        results = search(doc_id=doc_id, query=q, k=1)
        score = results[0]["score"] if results else 0.0
        covered_scores.append(score)
        if score >= threshold:
            passes = "YES (Passed)"
        else:
            passes = "NO (Wrong Filter)"
            covered_filtered_count += 1
        q_disp = (q[:52] + "...") if len(q) > 55 else q
        print(f"{idx:<3} | {'Covered':<12} | {q_disp:<55} | {score:<11.4f} | {passes:<15}")
        idx += 1

    print(sep)

    # Run not covered questions
    not_covered_passed_count = 0
    for q in NOT_COVERED_QUESTIONS:
        results = search(doc_id=doc_id, query=q, k=1)
        score = results[0]["score"] if results else 0.0
        not_covered_scores.append(score)
        if score >= threshold:
            passes = "YES (To LLM)"
            not_covered_passed_count += 1
        else:
            passes = "NO (Filtered)"
        q_disp = (q[:52] + "...") if len(q) > 55 else q
        print(f"{idx:<3} | {'Not Covered':<12} | {q_disp:<55} | {score:<11.4f} | {passes:<15}")
        idx += 1

    print(sep)

    # Summary Statistics
    min_covered = min(covered_scores) if covered_scores else 0.0
    max_covered = max(covered_scores) if covered_scores else 0.0
    min_not_covered = min(not_covered_scores) if not_covered_scores else 0.0
    max_not_covered = max(not_covered_scores) if not_covered_scores else 0.0

    print("\n--- Calibration Summary ---")
    print(f"Covered Questions (Total {len(COVERED_QUESTIONS)}):")
    print(f"  - Score Range: [{min_covered:.4f} .. {max_covered:.4f}]")
    print(f"  - Wrongly filtered at threshold {threshold:.4f}: {covered_filtered_count} / {len(COVERED_QUESTIONS)}")
    print(f"Not-Covered Questions (Total {len(NOT_COVERED_QUESTIONS)}):")
    print(f"  - Score Range: [{min_not_covered:.4f} .. {max_not_covered:.4f}]")
    print(f"  - Passing pre-filter to LLM at threshold {threshold:.4f}: {not_covered_passed_count} / {len(NOT_COVERED_QUESTIONS)}")

    print("\nPre-Filter Design Architecture:")
    print(f"  The threshold ({threshold:.4f}) acts strictly as a cheap, high-recall pre-filter.")
    print("  It immediately discards completely irrelevant queries (e.g. world facts, unrelated domains)")
    print("  without making API calls. For borderline or nearby ML topics that pass the pre-filter,")
    print("  the LLM acts as the definitive arbiter and responds with the standardized refusal sentence:")
    print("  'The paper does not appear to cover this.'")
    print("---------------------------\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate retrieval relevance threshold for an academic paper."
    )
    parser.add_argument("doc_id", help="Document ID (12-character hash) of the ingested paper")
    parser.add_argument(
        "--threshold",
        type=float,
        default=RELEVANCE_THRESHOLD,
        help=f"Current relevance threshold to evaluate against (default: {RELEVANCE_THRESHOLD})",
    )
    args = parser.parse_args()

    calibrate(doc_id=args.doc_id, threshold=args.threshold)


if __name__ == "__main__":
    main()
