import sys
import io
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Set stdout to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.agents.graph import ChatSession

print("=== FIX 6 TEST 1: Vague Question (Low Initial Retrieval) ===")
session = ChatSession("bdfaa68d8984")
res1 = session.ask("the special trick that prevents looking forward during generation")
print(f"Intent: {res1.intent}")
print(f"Trace: {' -> '.join(res1.trace)}")
print(f"Calls: {res1.llm_calls}")
print(f"Answer: {res1.answer[:250]}...")

print("\n=== FIX 6 TEST 2: Question Not Covered by Paper (Refusal/Exhaust) ===")
session2 = ChatSession("bdfaa68d8984")
res2 = session2.ask("what accuracy does BERT get on GLUE?")
print(f"Intent: {res2.intent}")
print(f"Trace: {' -> '.join(res2.trace)}")
print(f"Calls: {res2.llm_calls}")
print(f"Answer: {res2.answer}")
