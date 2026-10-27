import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from unittest.mock import patch, MagicMock
from app.agents.graph import ChatSession

with patch("app.llm.get_groq_client") as mgc:
    m = MagicMock()
    m.choices = [MagicMock(message=MagicMock(content='{"intent": "off_topic", "standalone_query": "q", "reason": "r"}'))]
    mgc.return_value.chat.completions.create.return_value = m
    s = ChatSession("doc")
    res = s.ask("hello")
    print("Turn calls for off_topic:", res.llm_calls)
    assert res.llm_calls == 1
    print("Verified successfully!")
