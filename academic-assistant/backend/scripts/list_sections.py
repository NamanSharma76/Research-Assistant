"""CLI tool to inspect distinct sections stored in ChromaDB for a document."""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.rag.retrieval import get_all_chunks


def list_sections(doc_id: str) -> None:
    """Prints every distinct section stored in ChromaDB for a doc, with chunk count and page range."""
    chunks = get_all_chunks(doc_id)
    if not chunks:
        print(f"No chunks found for document '{doc_id}'.")
        return

    # Aggregate by section
    sections = defaultdict(lambda: {"count": 0, "pages": []})
    for c in chunks:
        sec = c.get("section", "Unknown")
        sections[sec]["count"] += 1
        sections[sec]["pages"].append(c.get("page_start", 1))
        sections[sec]["pages"].append(c.get("page_end", 1))

    print(f"\nSections stored for doc_id: {doc_id} (Total Chunks: {len(chunks)})\n")
    header = f"{'Section':<35} | {'Chunks':<8} | {'Page Range':<12}"
    print(header)
    print("-" * len(header))

    for sec, data in sections.items():
        min_p = min(data["pages"])
        max_p = max(data["pages"])
        p_range = f"p. {min_p}" if min_p == max_p else f"pp. {min_p}-{max_p}"
        print(f"{sec:<35} | {data['count']:<8} | {p_range:<12}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List distinct sections stored in ChromaDB for a doc_id."
    )
    parser.add_argument("doc_id", help="Document ID (12-character hash) of the ingested paper")
    args = parser.parse_args()

    list_sections(args.doc_id)


if __name__ == "__main__":
    main()
