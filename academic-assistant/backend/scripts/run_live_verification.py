import sys
import io
import time
import httpx
from pathlib import Path

# Force utf-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"
PDF_PATH = r"C:\Projects\Research Assistant\academic-assistant\backend\data\uploads\attention_is_all_you_need.pdf"

client = httpx.Client(base_url=BASE_URL, timeout=120.0)

print("==========================================")
print("1. Register + Login User 1")
print("=========================================="
)
try:
    reg_res = client.post("/auth/register", json={"email": "live_user1@example.com", "password": "SuperSecretPassword123"})
    print("Register Status:", reg_res.status_code, reg_res.json())
except Exception as e:
    print("Register info:", e)

login_res = client.post("/auth/login", json={"email": "live_user1@example.com", "password": "SuperSecretPassword123"})
print("Login Status:", login_res.status_code)
token1 = login_res.json()["access_token"]
print("Token (truncated):", token1[:28] + "...")
headers1 = {"Authorization": f"Bearer {token1}"}

print("\n==========================================")
print("2. Upload attention_is_all_you_need.pdf")
print("==========================================")
with open(PDF_PATH, "rb") as f:
    upload_res = client.post("/documents", files={"file": ("attention_is_all_you_need.pdf", f, "application/pdf")}, headers=headers1)

print("Upload Status:", upload_res.status_code)
doc_data = upload_res.json()
print("Document Upload Result:", doc_data)
doc_id = doc_data["id"]

print(f"\nPolling GET /documents/{doc_id} until ready...")
for i in range(40):
    poll_res = client.get(f"/documents/{doc_id}", headers=headers1)
    status_val = poll_res.json().get("status")
    num_chunks = poll_res.json().get("num_chunks")
    print(f"[{i+1}] Status: {status_val} | num_chunks: {num_chunks}")
    if status_val in ("ready", "failed"):
        break
    time.sleep(2)

print("\n==========================================")
print("3. Create Conversation & Send 3 Messages")
print("==========================================")
conv_res = client.post(f"/documents/{doc_id}/conversations", json={"title": "Attention Live Chat"}, headers=headers1)
print("Conversation Create Status:", conv_res.status_code)
conv_data = conv_res.json()
conv_id = conv_data["id"]
print("Conversation ID:", conv_id)

questions = [
    "What dataset did they use for translation?",
    "explain that more simply",
    "what is the capital of France?",
]

for q in questions:
    print(f"\n--- User: {q} ---")
    msg_res = client.post(f"/conversations/{conv_id}/messages", json={"content": q}, headers=headers1)
    print("Status:", msg_res.status_code)
    m = msg_res.json()
    print("Intent:", m.get("intent"))
    print("Trace:", " -> ".join(m.get("trace") or []))
    print(f"LLM Calls: {m.get('llm_calls')} | Time: {m.get('seconds')}s")
    content = m.get("content", "")
    preview = content[:250] + "..." if len(content) > 250 else content
    print("Response:", preview)

print("\n==========================================")
print("4. GET notes and GET outline?slides=6")
print("==========================================")
notes_res = client.get(f"/documents/{doc_id}/notes", headers=headers1)
print("Notes Status:", notes_res.status_code)
notes_data = notes_res.json()
problem_text = notes_data.get("problem", {}).get("text", "")
print("Notes Problem preview:", problem_text[:160] + "...")

outline_res = client.get(f"/documents/{doc_id}/outline?slides=6", headers=headers1)
print("Outline Status:", outline_res.status_code)
outline_data = outline_res.json()
print("Outline Title:", outline_data.get("title"))
print("Slides Count:", len(outline_data.get("slides", [])))
for s in outline_data.get("slides", []):
    print(f"  Slide {s.get('slide_number')}: {s.get('title')} [Source: {s.get('source')}]")

print("\n==========================================")
print("5. Register Second User & Test 404 Ownership")
print("==========================================")
try:
    client.post("/auth/register", json={"email": "live_user2@example.com", "password": "User2Password123"})
except Exception:
    pass
login2 = client.post("/auth/login", json={"email": "live_user2@example.com", "password": "User2Password123"})
token2 = login2.json()["access_token"]
headers2 = {"Authorization": f"Bearer {token2}"}

user2_doc_res = client.get(f"/documents/{doc_id}", headers=headers2)
print("User 2 accessing User 1 document -> Status Code:", user2_doc_res.status_code, user2_doc_res.json())

user2_conv_res = client.get(f"/conversations/{conv_id}/messages", headers=headers2)
print("User 2 accessing User 1 conversation -> Status Code:", user2_conv_res.status_code, user2_conv_res.json())

user2_notes_res = client.get(f"/documents/{doc_id}/notes", headers=headers2)
print("User 2 accessing User 1 notes -> Status Code:", user2_notes_res.status_code, user2_notes_res.json())

print("\n==========================================")
print("6. Delete Document & Check Status")
print("==========================================")
del_res = client.delete(f"/documents/{doc_id}", headers=headers1)
print("Delete Document Status:", del_res.status_code)

after_del = client.get(f"/documents/{doc_id}", headers=headers1)
print("GET Document after deletion Status:", after_del.status_code, after_del.json())
