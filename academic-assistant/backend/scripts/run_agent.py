import argparse
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend directory to sys.path so app modules import properly
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.llm import get_call_count, reset_call_count, get_call_stats, reset_call_stats
from app.agents import cache
from app.agents.summarizer import summarize_section, summarize_document
from app.agents.notes import generate_notes
from app.agents.outline import generate_outline
from app.agents.render import render_summary, render_notes, render_outline


def print_stats(start_time: float, cache_hit: bool) -> None:
    elapsed = time.time() - start_time
    stats = get_call_stats()
    calls = stats["call_count"]
    retries = stats["rate_limit_retries"]
    slowest_sec = stats["slowest_seconds"]
    slowest_m = stats["slowest_model"]
    print(f"\n--- [LLM calls: {calls} | 429 retries: {retries} | slowest call: {slowest_sec:.2f} s ({slowest_m}) | Time: {elapsed:.2f} s] ---")


def handle_summarize(args) -> None:
    reset_call_stats()
    start_time = time.time()
    doc_id = args.doc_id

    if args.section:
        print(f"=== SECTION SUMMARY: {args.section} ({doc_id}) ===")
        summary = summarize_section(doc_id, args.section)
        print(summary)
        print_stats(start_time, cache_hit=False)
        return

    cached_before = cache.get(doc_id, "summary") is not None and not args.force
    result = summarize_document(doc_id, force=args.force)
    print(render_summary(result, doc_id))
    print_stats(start_time, cache_hit=cached_before)


def handle_notes(args) -> None:
    reset_call_stats()
    start_time = time.time()
    doc_id = args.doc_id

    cached_before = cache.get(doc_id, "notes") is not None and not args.force
    notes = generate_notes(doc_id, force=args.force)
    print(render_notes(notes, doc_id))
    print_stats(start_time, cache_hit=cached_before)


def handle_outline(args) -> None:
    reset_call_stats()
    start_time = time.time()
    doc_id = args.doc_id
    num_slides = args.slides

    cache_kind = f"outline_{num_slides}"
    cached_before = cache.get(doc_id, cache_kind) is not None and not args.force
    outline = generate_outline(doc_id, num_slides=num_slides, force=args.force)
    print(render_outline(outline, num_slides))
    print_stats(start_time, cache_hit=cached_before)


def main():
    parser = argparse.ArgumentParser(description="Run research assistant agents on ingested documents.")
    parser.add_argument("doc_id", help="Document ID")
    parser.add_argument("command", choices=["summarize", "notes", "outline"], help="Agent command to run")
    parser.add_argument("--section", type=str, default=None, help="Specific section name to summarize")
    parser.add_argument("--slides", type=int, default=10, help="Number of slides (5-15, default 10)")
    parser.add_argument("--force", action="store_true", help="Bypass cache and regenerate")

    args = parser.parse_args()

    if args.command == "summarize":
        handle_summarize(args)
    elif args.command == "notes":
        handle_notes(args)
    elif args.command == "outline":
        handle_outline(args)


if __name__ == "__main__":
    main()
