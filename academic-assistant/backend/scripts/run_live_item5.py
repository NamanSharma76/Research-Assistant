import json
import httpx

with open("academic-assistant/backend/data/live_state.json", "r") as f:
    state = json.load(f)

tokenB = state["tokenB"]
docIdA = state["docId"]
convIdA = state["convId"]
base_url = "http://127.0.0.1:8000"
headers = {"Authorization": f"Bearer {tokenB}"}

endpoints = [
    ("GET /documents/{A_id}", "GET", f"{base_url}/documents/{docIdA}", None),
    ("DELETE /documents/{A_id}", "DELETE", f"{base_url}/documents/{docIdA}", None),
    ("GET /documents/{A_id}/sections", "GET", f"{base_url}/documents/{docIdA}/sections", None),
    ("GET /documents/{A_id}/summary", "GET", f"{base_url}/documents/{docIdA}/summary", None),
    ("GET /documents/{A_id}/notes", "GET", f"{base_url}/documents/{docIdA}/notes", None),
    ("GET /documents/{A_id}/outline", "GET", f"{base_url}/documents/{docIdA}/outline", None),
    ("POST /documents/{A_id}/conversations", "POST", f"{base_url}/documents/{docIdA}/conversations", {"title": "Intruder"}),
    ("GET /conversations/{C1}/messages", "GET", f"{base_url}/conversations/{convIdA}/messages", None),
    ("POST /conversations/{C1}/messages", "POST", f"{base_url}/conversations/{convIdA}/messages", {"content": "Intruder message"}),
    ("DELETE /conversations/{C1}", "DELETE", f"{base_url}/conversations/{convIdA}", None),
]

print("==================================================")
print("ITEM 5: User B Isolation Tests (Expect 404 on all)")
print("==================================================")

client = httpx.Client(timeout=10.0)
for label, method, url, json_body in endpoints:
    if method == "GET":
        resp = client.get(url, headers=headers)
    elif method == "DELETE":
        resp = client.delete(url, headers=headers)
    elif method == "POST":
        resp = client.post(url, headers=headers, json=json_body)
    print(f"[{label}] -> HTTP {resp.status_code} | Body: {resp.text}")
