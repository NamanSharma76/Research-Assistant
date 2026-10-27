import json
import sys
import httpx

sys.stdout.reconfigure(encoding="utf-8")

with open("academic-assistant/backend/data/live_state.json", "r") as f:
    state = json.load(f)

tokenA = state["tokenA"]
docIdA = state["docId"]
base_url = "http://127.0.0.1:8000"
headers = {"Authorization": f"Bearer {tokenA}"}

print("==================================================")
print("ITEM 6: User A Requests Artifacts (Notes, Outline, Summary)")
print("==================================================")

client = httpx.Client(timeout=300.0)

# 1. Notes
print("\n--> GET /documents/{id}/notes")
res_notes = client.get(f"{base_url}/documents/{docIdA}/notes", headers=headers)
print(f"Status: {res_notes.status_code}")
notes_data = res_notes.json()
print("Notes JSON Keys:", list(notes_data.keys()))
print("Problem text preview:", notes_data.get("problem", {}).get("text", "")[:200])
print("Method text preview:", notes_data.get("method", {}).get("text", "")[:200])
print("Key Terms count:", len(notes_data.get("key_terms", [])))
print("Revision Questions count:", len(notes_data.get("revision_questions", [])))

# 2. Outline with 6 slides
print("\n--> GET /documents/{id}/outline?slides=6")
res_outline = client.get(f"{base_url}/documents/{docIdA}/outline?slides=6", headers=headers)
print(f"Status: {res_outline.status_code}")
outline_data = res_outline.json()
print(f"Total slides returned: {len(outline_data.get('slides', []))}")
for i, slide in enumerate(outline_data.get("slides", []), 1):
    print(f"\n--- Slide {i}: {slide.get('title')} ---")
    print(f"Source: {slide.get('source')}")
    print(f"Source Fields: {slide.get('source_fields')}")
    for b in slide.get("bullets", []):
        print(f"  * {b}")

# 3. Summary
print("\n--> GET /documents/{id}/summary")
res_summary = client.get(f"{base_url}/documents/{docIdA}/summary", headers=headers)
print(f"Status: {res_summary.status_code}")
summary_data = res_summary.json()
print(f"Summary Document ID: {summary_data.get('doc_id')}")
print("Executive Summary Preview:\n", summary_data.get("executive_summary", "")[:350])
print("Sections summarized:", list(summary_data.get("sections", {}).keys()))
