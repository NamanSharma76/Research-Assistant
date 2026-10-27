import sys
import io
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Set stdout to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.agents.graph import ChatSession

def truncate_display(text: str, max_chars: int = 600) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} characters]"
    return text

def main():
    doc_id = "bdfaa68d8984"
    session = ChatSession(doc_id=doc_id)
    questions_file = Path(__file__).resolve().parent / "scripted_conversation.txt"
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"=== Academic Research Assistant Chat (Doc ID: {doc_id}) ===")
    for idx, q in enumerate(questions, 1):
        print(f"\n[Turn {idx}] User: {q}")
        res = session.ask(q)
        trace_path = " -> ".join(res.trace)
        print(f"Intent: {res.intent}")
        print(f"Standalone Query: {res.standalone_query}")
        print(f"Path: {trace_path}")
        print(f"LLM Calls: {res.llm_calls} | Time: {res.seconds:.2f}s")
        print("--- Response ---")
        print(truncate_display(res.answer))

if __name__ == "__main__":
    main()
