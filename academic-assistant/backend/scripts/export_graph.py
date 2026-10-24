"""Exports the LangGraph orchestration workflow as a Mermaid diagram."""
from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.agents.graph import compiled_app


def main():
    mermaid_text = compiled_app.get_graph().draw_mermaid()
    docs_dir = backend_dir.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_file = docs_dir / "graph.mmd"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(mermaid_text)

    print(f"Exported Mermaid graph to: {out_file}\n")
    print(mermaid_text)


if __name__ == "__main__":
    main()
