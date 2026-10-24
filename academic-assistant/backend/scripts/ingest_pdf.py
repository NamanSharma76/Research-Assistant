"""CLI tool for ingesting academic research paper PDFs into ChromaDB."""
import argparse
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.rag.ingest import ingest_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest an academic research paper PDF into ChromaDB."
    )
    parser.add_argument("pdf_path", help="Path to the research paper PDF file")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion by deleting existing chunks for this doc_id first",
    )
    args = parser.parse_args()

    pdf_file = Path(args.pdf_path).resolve()
    if not pdf_file.exists():
        print(f"Error: File not found at '{pdf_file}'", file=sys.stderr)
        sys.exit(1)

    print(f"Ingesting: {pdf_file.name} {'(force re-index)' if args.force else ''}...")
    result = ingest_pdf(str(pdf_file), force=args.force)

    print("\n--- Ingestion Result ---")
    print(f"Status:       {result['status']}")
    print(f"Doc ID:       {result['doc_id']}")
    print(f"Filename:     {result['filename']}")
    print(f"Total Chunks: {result['chunks_count']}")
    print("------------------------\n")
    if result["status"] == "already_indexed":
        print("Note: Document was already indexed. No duplicate chunks were added.")


if __name__ == "__main__":
    main()
