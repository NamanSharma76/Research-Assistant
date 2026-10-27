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
print("7. Post-Restart Persistence & New Turn")
print("==========================================")
login_res = client.post("/auth/login", json={"email": "live_user1@example.com", "password": "SuperSecretPassword123"})
token1 = login_res.json()["access_token"]
headers1 = {"Authorization": f"Bearer {token1}"}

# Re-upload document for user1
with open(PDF_PATH, "rb") as f:
    upload_res = client.post("/documents", files={"file": ("attention_is_all_you_need.pdf", f, "application/pdf")}, headers=headers1)
doc_id = upload_res.json()["id"]

# Poll ready
for _ in range(30):
    poll = client.get(f"/documents/{doc_id}", headers=headers1)
    if poll.json().get("status") == "ready":
        break
    time.sleep(1)

# Create conversation
conv = client.post(f"/documents/{doc_id}/conversations", json={"title": "Persistent Chat"}, headers=headers1).json()
conv_id = conv["id"]

# Send message 1
msg1 = client.post(f"/conversations/{conv_id}/messages", json={"content": "What is the Transformer?"}, headers=headers1).json()
print("First message answer preview:", msg1.get("content", "")[:120] + "...")

# Get messages list
list_before = client.get(f"/conversations/{conv_id}/messages", headers=headers1).json()
print(f"Messages count before new turn: {len(list_before)}")

# Send message 2 (new turn)
msg2 = client.post(f"/conversations/{conv_id}/messages", json={"content": "what dataset was used for translation?"}, headers=headers1).json()
print("Second message intent:", msg2.get("intent"))
print("Second message answer preview:", msg2.get("content", "")[:120] + "...")

# Get messages list after
list_after = client.get(f"/conversations/{conv_id}/messages", headers=headers1).json()
print(f"Messages count after new turn: {len(list_after)}")
for idx, m in enumerate(list_after, 1):
    print(f"  [{idx}] {m['role']}: {m['content'][:60]}...")
