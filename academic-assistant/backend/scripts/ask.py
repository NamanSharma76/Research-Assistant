"""CLI tool for cited Q&A over ingested academic papers."""
import argparse
import sys
from pathlib import Path

# Force UTF-8 stdout encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.rag.retrieval import search
from app.rag.qa import answer_question
from app.config import RELEVANCE_THRESHOLD


def print_sources_used(chunks: list) -> None:
    """Prints the SOURCES USED block matching excerpt numbers sent to LLM."""
    print("SOURCES USED")
    for idx, c in enumerate(chunks, 1):
        p_str = f"p. {c['page_start']}" if c['page_start'] == c['page_end'] else f"pp. {c['page_start']}-{c['page_end']}"
        print(f"  [{idx}] {c['section']}, {p_str} (score: {c['score']:.4f})")


def ask(doc_id: str, question: str, k: int = 5, threshold: float = RELEVANCE_THRESHOLD) -> None:
    """Performs cited Q&A on an ingested paper."""
    print(f"Searching document '{doc_id}' for query: '{question}'...")
    chunks = search(doc_id=doc_id, query=question, k=k)

    if not chunks:
        print("\nThe paper does not appear to cover this.")
        return

    best_score = chunks[0]["score"]
    print(f"Best similarity score: {best_score:.4f} (Threshold: {threshold:.4f})\n")

    if best_score < threshold:
        print("The paper does not appear to cover this.")
        return

    # Relevant chunks above threshold
    relevant_chunks = [c for c in chunks if c["score"] >= threshold]
    if not relevant_chunks:
        print("The paper does not appear to cover this.")
        return

    try:
        qa_res = answer_question(doc_id=doc_id, question=question, chunks=relevant_chunks)
    except ValueError as e:
        # GROQ_API_KEY not configured: print notice and summarize grounded excerpts
        print(f"[Notice: {e}]")
        print("\n=== Excerpt Citations (LLM Key not set) ===")
        print("Retrieved grounded context from paper ready for LLM synthesis:")
        for idx, c in enumerate(relevant_chunks, 1):
            p_str = f"p. {c['page_start']}" if c['page_start'] == c['page_end'] else f"pp. {c['page_start']}-{c['page_end']}"
            print(f"\n[{idx}] [{c['section']}, {p_str}] (score: {c['score']:.4f})")
            print(f"    {c['text'][:250]}...")
        return
    except Exception as e:
        print(f"\nError calling LLM: {e}", file=sys.stderr)
        return

    raw_answer = qa_res["raw_answer"]
    display_answer = qa_res["answer"]
    cit_result = qa_res["citation_check"]
    n_cited = len(cit_result["cited"])
    m_supported = len(cit_result["supported"])
    k_unsupported = len(cit_result["unsupported"])

    print("============================== RAW ANSWER ==============================")
    print(raw_answer.strip())
    print("========================================================================\n")

    if qa_res["refused"]:
        print("The paper does not appear to cover this. (judged by LLM from retrieved excerpts)\n")
        print_sources_used(relevant_chunks)
        return

    print("============================== ANSWER ==============================")
    print(display_answer.strip())
    print("====================================================================\n")

    # Print citation check line
    print(f"Citation check: {n_cited} cited, {m_supported} supported, {k_unsupported} unsupported")
    if k_unsupported > 0:
        print("Warning: Unsupported citations detected:")
        for un in cit_result["unsupported"]:
            p_str = f"p. {un['page_start']}" if un['page_start'] == un['page_end'] else f"pp. {un['page_start']}-{un['page_end']}"
            reason_info = f" ({un['reason']})" if un.get("reason") else ""
            print(f"  - [{un['section']}, {p_str}]{reason_info}")
    print()

    # Print SOURCES USED block
    print_sources_used(relevant_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question about an ingested academic research paper."
    )
    parser.add_argument("doc_id", help="Document ID (12-character hash) of the ingested paper")
    parser.add_argument("question", help="Question to ask about the paper")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=RELEVANCE_THRESHOLD,
        help=f"Relevance score threshold (default: {RELEVANCE_THRESHOLD})",
    )
    args = parser.parse_args()

    ask(doc_id=args.doc_id, question=args.question, k=args.k, threshold=args.threshold)


if __name__ == "__main__":
    main()
