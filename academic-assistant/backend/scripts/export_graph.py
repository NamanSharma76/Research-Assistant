"""Exports the LangGraph orchestration workflow as a Mermaid diagram."""
import re
from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.agents.graph import compiled_app


def clean_mermaid(text: str) -> str:
    """Cleans mermaid diagram: strips frontmatter/config, removes trailing semicolons, ensures standard header."""
    # Remove config / frontmatter block if present
    text = re.sub(r"^---\n[\s\S]*?\n---\n?", "", text, flags=re.MULTILINE).strip()
    # Normalize header to graph TD
    text = re.sub(r"^(graph|flowchart)\s+[A-Z]+;?", "flowchart TD", text)
    # Remove trailing semicolons from lines
    lines = []
    for line in text.splitlines():
        lines.append(line.rstrip(";"))
    return "\n".join(lines).strip() + "\n"


def main():
    raw_mermaid = compiled_app.get_graph().draw_mermaid()
    cleaned_mermaid = clean_mermaid(raw_mermaid)

    docs_dir = backend_dir.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    mmd_file = docs_dir / "graph.mmd"
    with open(mmd_file, "w", encoding="utf-8") as f:
        f.write(cleaned_mermaid)

    md_file = docs_dir / "graph.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Academic Assistant - LangGraph Workflow\n\n```mermaid\n{cleaned_mermaid}```\n")

    print(f"Exported Mermaid graph to: {mmd_file} and {md_file}\n")
    print(cleaned_mermaid)


if __name__ == "__main__":
    main()
