import json
import httpx
import sys
import os

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, backend_path)

from app.rag.ingest import get_chroma_client
from app.config import CHROMA_COLLECTION_NAME

def get_chroma_count(doc_id="bdfaa68d8984"):
    client = get_chroma_client()
    collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    res = collection.get(where={"doc_id": doc_id})
    return len(res["ids"])

with open("academic-assistant/backend/data/live_state.json", "r") as f:
    state = json.load(f)

tokenA = state["tokenA"]
tokenB = state["tokenB"]
docIdA = state["docId"]
base_url = "http://127.0.0.1:8000"
headersA = {"Authorization": f"Bearer {tokenA}"}
headersB = {"Authorization": f"Bearer {tokenB}"}
pdf_path = "academic-assistant/backend/data/uploads/attention_is_all_you_need.pdf"

print("==================================================")
print("ITEM 7: Shared Document Deletion & Chroma Retention")
print("==================================================")

# Step 7.1: Count ChromaDB chunks for bdfaa68d8984
c1 = get_chroma_count("bdfaa68d8984")
print(f"Step 7.1: ChromaDB chunk count for bdfaa68d8984 before deletion: {c1}")

# Step 7.2: User A deletes document
client = httpx.Client(timeout=30.0)
res_del = client.delete(f"{base_url}/documents/{docIdA}", headers=headersA)
print(f"Step 7.2: DELETE /documents/{docIdA} status: {res_del.status_code} (Expected 204)")

# Step 7.3: Count again -> should be 0 since no other user has this doc_id
c2 = get_chroma_count("bdfaa68d8984")
print(f"Step 7.3: ChromaDB chunk count for bdfaa68d8984 after sole user deletion: {c2} (Expected 0)")

# Step 7.4: Upload as User A and User B
print("\nStep 7.4: Re-uploading document as User A and User B...")
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

res_up_A = client.post(f"{base_url}/documents", headers=headersA, files={"file": ("paper.pdf", pdf_bytes, "application/pdf")})
doc_id_A2 = res_up_A.json()["id"]
print(f"User A upload response: HTTP {res_up_A.status_code}, DB doc_id: {doc_id_A2}")

res_up_B = client.post(f"{base_url}/documents", headers=headersB, files={"file": ("paper.pdf", pdf_bytes, "application/pdf")})
doc_id_B2 = res_up_B.json()["id"]
print(f"User B upload response: HTTP {res_up_B.status_code}, DB doc_id: {doc_id_B2}")

# Poll until both ready
import time
for _ in range(20):
    time.sleep(1)
    stA = client.get(f"{base_url}/documents/{doc_id_A2}", headers=headersA).json()
    stB = client.get(f"{base_url}/documents/{doc_id_B2}", headers=headersB).json()
    if stA.get("status") == "ready" and stB.get("status") == "ready":
        break

c_shared = get_chroma_count("bdfaa68d8984")
print(f"Both ready. Chroma chunk count shared: {c_shared} (Expected 20)")

# Step 7.5: User A deletes their document
res_del_A = client.delete(f"{base_url}/documents/{doc_id_A2}", headers=headersA)
print(f"Step 7.5: User A DELETE /documents/{doc_id_A2} status: {res_del_A.status_code} (Expected 204)")

c_after_A_del = get_chroma_count("bdfaa68d8984")
print(f"Step 7.5: Chroma count after User A deletion (User B still references it): {c_after_A_del} (Expected > 0, namely 20)")

# Step 7.6: User B deletes their document
res_del_B = client.delete(f"{base_url}/documents/{doc_id_B2}", headers=headersB)
print(f"Step 7.6: User B DELETE /documents/{doc_id_B2} status: {res_del_B.status_code} (Expected 204)")

c_after_B_del = get_chroma_count("bdfaa68d8984")
print(f"Step 7.6: Chroma count after User B deletion (last reference removed): {c_after_B_del} (Expected 0)")
