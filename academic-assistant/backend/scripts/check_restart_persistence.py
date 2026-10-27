import sys
import io
import httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

client = httpx.Client(base_url="http://127.0.0.1:8000")
token = client.post("/auth/login", json={"email": "live_user1@example.com", "password": "SuperSecretPassword123"}).json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
convs = client.get("/conversations", headers=headers).json()
print("Conversations count:", len(convs))
if convs:
    c_id = convs[0]["id"]
    msgs = client.get(f"/conversations/{c_id}/messages", headers=headers).json()
    print("Messages preserved across restart:", len(msgs))
    for m in msgs:
        print(f"  [{m['role']}]: {m['content'][:70]}...")

