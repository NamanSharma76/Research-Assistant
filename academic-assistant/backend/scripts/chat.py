"""Interactive and scripted chat CLI tool with LangGraph routing and memory."""
import argparse
import sys
from pathlib import Path

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.agents.graph import ChatSession


def truncate_display(text: str, max_chars: int = 600) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} characters]"
    return text


def run_turn(session: ChatSession, user_msg: str, trace_flag: bool) -> None:
    print(f"\nUser: {user_msg}")
    res = session.ask(user_msg)

    if trace_flag:
        trace_path = " -> ".join(res.trace)
        print(f"Intent: {res.intent}")
        print(f"Standalone Query: {res.standalone_query}")
        print(f"Path: {trace_path}")
        print(f"LLM Calls: {res.llm_calls} | Time: {res.seconds:.2f}s")
        print("--- Response ---")

    print(truncate_display(res.answer))


def main():
    parser = argparse.ArgumentParser(description="Academic Assistant conversational chat interface.")
    parser.add_argument("doc_id", help="Document ID of the ingested research paper")
    parser.add_argument("--trace", action="store_true", help="Print routing intent, query, path, and stats per turn")
    args = parser.parse_args()

    session = ChatSession(doc_id=args.doc_id)

    # Check if input is piped from stdin
    if not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if line:
                run_turn(session, line, args.trace)
        return

    # Interactive mode
    print(f"=== Academic Research Assistant Chat (Doc ID: {args.doc_id}) ===")
    print("Type your message and press Enter. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_msg = input("\nYou: ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            run_turn(session, user_msg, args.trace)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break


if __name__ == "__main__":
    main()
