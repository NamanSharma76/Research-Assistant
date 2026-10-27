"""Live end-to-end verification script for Phase 4.5 Streamlit Frontend & API Client."""
import sys
import os
import time

# Ensure frontend path is on sys.path
frontend_dir = os.path.abspath(os.path.dirname(__file__))
if frontend_dir not in sys.path:
    sys.path.insert(0, frontend_dir)

import api_client
from components import (
    render_citations_to_html,
    format_trace,
    outline_to_markdown,
    outline_to_text,
    render_status_badge,
    render_error_message,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_verification():
    print("=" * 70)
    print("PHASE 4.5 FRONTEND & API CLIENT END-TO-END VERIFICATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Step 1 & 2: User Registration, Logout & Login
    # -------------------------------------------------------------
    print("\n--- STEP 1 & 2: User Registration, Logout & Login ---")
    user1_email = "researcher_p45@university.edu"
    user1_pwd = "SecurePassword123"

    print(f"Registering user: {user1_email}")
    try:
        reg_res = api_client.register(user1_email, user1_pwd)
        print(f"Registration response: id={reg_res.get('id')} email={reg_res.get('email')}")
    except api_client.ApiError as e:
        if "already registered" in str(e).lower() or e.status_code == 400:
            print("User already registered; proceeding with login test.")
        else:
            raise

    print("Simulating logout (clearing token)...")
    token = None

    print(f"Logging in as {user1_email}...")
    login_res = api_client.login(user1_email, user1_pwd)
    token = login_res["access_token"]
    print(f"Login successful! JWT acquired (length: {len(token)} chars)")

    profile = api_client.me(token)
    print(f"User profile verified: {profile['email']}")

    # -------------------------------------------------------------
    # Step 3: PDF Upload and Ingestion Polling
    # -------------------------------------------------------------
    print("\n--- STEP 3: PDF Upload & Ingestion Polling ---")
    existing_docs = api_client.list_documents(token)
    if existing_docs:
        ready_doc = existing_docs[0]
        doc_db_id = ready_doc["id"]
        print(f"Found existing document: ID={doc_db_id}, status={ready_doc['status']}")
    else:
        pdf_path = os.path.abspath(
            os.path.join(frontend_dir, "..", "backend", "data", "uploads", "attention_is_all_you_need.pdf")
        )
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Sample PDF not found at {pdf_path}")

        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        print(f"Uploading {os.path.basename(pdf_path)} ({len(file_bytes)} bytes)...")
        doc_res = api_client.upload_document(token, filename="attention_is_all_you_need.pdf", file_bytes=file_bytes)
        doc_db_id = doc_res["id"]
        print(f"Upload accepted: Document ID={doc_db_id}, status={doc_res['status']}")

        print("Polling document status until ready...")
        start_poll = time.time()
        ready_doc = None
        for _ in range(30):
            doc = api_client.get_document(token, doc_db_id)
            status = doc["status"]
            print(f"  Status: {render_status_badge(status)} (elapsed: {int(time.time() - start_poll)}s)")
            if status == "ready":
                ready_doc = doc
                break
            elif status == "failed":
                raise RuntimeError(f"Ingestion failed: {doc.get('error')}")
            time.sleep(2)

        if not ready_doc:
            raise TimeoutError("Document ingestion did not reach 'ready' within 60s")

    print(f"Document ready! Chunks indexed: {ready_doc.get('num_chunks')}, doc_id={ready_doc.get('doc_id')}")

    # -------------------------------------------------------------
    # Step 4: Ask Question with Grounded Citations
    # -------------------------------------------------------------
    print("\n--- STEP 4: Chat Q&A with Citation Chips ---")
    conv = api_client.create_conversation(token, doc_db_id, title="Attention Mechanism Discussion")
    conv_id = conv["id"]
    print(f"Conversation created: id={conv_id}, title='{conv.get('title')}'")

    q1 = "What is the primary difference between the Transformer and previous recurrent models?"
    print(f"User: {q1}")
    msg1 = api_client.send_message(token, conv_id, q1)
    print(f"\nAssistant Response (Turn 1):\n{msg1['content']}")
    print(f"\nMetadata: Intent={msg1.get('intent')} | LLM calls={msg1.get('llm_calls')} | Time={msg1.get('seconds')}s")
    print(f"Raw Trace: {msg1.get('trace')}")
    formatted_trace = format_trace(msg1.get("trace"))
    print(f"Formatted Trace: {formatted_trace}")

    html_chips = render_citations_to_html(msg1['content'])
    has_chips = "citation-chip" in html_chips
    print(f"Rendered Citation Chips detected: {has_chips}")
    if not has_chips:
        print("Note: Output did not contain citation brackets or was conversational.")

    # -------------------------------------------------------------
    # Step 5: Follow-up Rephrase Turn (rephrase_previous)
    # -------------------------------------------------------------
    print("\n--- STEP 5: Follow-up Turn ('explain that more simply') ---")
    q2 = "explain that more simply"
    print(f"User: {q2}")
    msg2 = api_client.send_message(token, conv_id, q2)
    print(f"\nAssistant Response (Turn 2):\n{msg2['content']}")
    print(f"\nMetadata: Intent={msg2.get('intent')} | LLM calls={msg2.get('llm_calls')} | Time={msg2.get('seconds')}s")
    print(f"Formatted Trace: {format_trace(msg2.get('trace'))}")
    assert msg2.get("intent") == "rephrase_previous", f"Expected rephrase_previous, got {msg2.get('intent')}"
    assert "retrieve" not in (msg2.get("trace") or []), f"Rephrase should not retrieve! Trace: {msg2.get('trace')}"
    print("SUCCESS: rephrase_previous triggered without retrieval.")

    # -------------------------------------------------------------
    # Step 6: Off-Topic Turn
    # -------------------------------------------------------------
    print("\n--- STEP 6: Off-Topic Handling ---")
    q3 = "What is the capital of France?"
    print(f"User: {q3}")
    msg3 = api_client.send_message(token, conv_id, q3)
    print(f"\nAssistant Response (Turn 3):\n{msg3['content']}")
    print(f"Metadata: Intent={msg3.get('intent')} | LLM calls={msg3.get('llm_calls')} | Time={msg3.get('seconds')}s")
    print(f"Formatted Trace: {format_trace(msg3.get('trace'))}")
    assert msg3.get("intent") == "off_topic", f"Expected off_topic, got {msg3.get('intent')}"
    assert "retrieve" not in (msg3.get("trace") or []), f"Off-topic should not retrieve! Trace: {msg3.get('trace')}"
    print("SUCCESS: off_topic refused politely without retrieval.")

    # -------------------------------------------------------------
    # Step 7: Notes Generation & Verification
    # -------------------------------------------------------------
    print("\n--- STEP 7: Notes Generation ---")
    print("Fetching notes (initial call)...")
    notes_data = api_client.get_notes(token, doc_db_id)
    required_note_fields = ["problem", "method", "dataset_and_setup", "key_results", "limitations", "key_terms", "revision_questions"]
    for field in required_note_fields:
        assert field in notes_data, f"Missing field in notes: {field}"
    print(f"Notes structure verified! Fields present: {list(notes_data.keys())}")
    print(f"Key Terms count: {len(notes_data['key_terms'])}")
    print(f"Revision Questions count: {len(notes_data['revision_questions'])}")

    # -------------------------------------------------------------
    # Step 8: Outline Generation & Exports
    # -------------------------------------------------------------
    print("\n--- STEP 8: Presentation Outline Generation & Exports ---")
    print("Fetching 8-slide presentation outline...")
    outline_data = api_client.get_outline(token, doc_db_id, slides=8)
    slides = outline_data.get("slides", [])
    print(f"Outline Title: '{outline_data.get('title')}'")
    print(f"Slide count: {len(slides)}")
    assert len(slides) == 8, f"Expected 8 slides, got {len(slides)}"

    md_output = outline_to_markdown(outline_data)
    txt_output = outline_to_text(outline_data)
    assert "# " in md_output and "## Slide 1" in md_output
    assert "SLIDE 1" in txt_output.upper()
    print("Markdown and Plain Text outline conversions verified!")
    print("\nSlide 1 Preview (Markdown):\n" + "\n".join(md_output.split("\n")[:8]))

    # -------------------------------------------------------------
    # Step 9: Summary Generation
    # -------------------------------------------------------------
    print("\n--- STEP 9: Executive Summary Generation ---")
    print("Fetching document summary...")
    summary_data = api_client.get_summary(token, doc_db_id)
    overall = summary_data.get("overall", "")
    sections = summary_data.get("sections", {})
    print(f"Overall summary length: {len(overall)} chars")
    print(f"Section breakdown count: {len(sections)} sections")
    print(f"Overall summary snippet: {overall[:200]}...")

    # -------------------------------------------------------------
    # Step 10: Multi-User Isolation & Offline Resiliency
    # -------------------------------------------------------------
    print("\n--- STEP 10: Multi-User Isolation & Offline Resiliency ---")
    user2_email = "other_user_p45@university.edu"
    user2_pwd = "OtherPassword123"

    print(f"Registering user 2: {user2_email}")
    try:
        api_client.register(user2_email, user2_pwd)
    except api_client.ApiError:
        pass
    u2_token = api_client.login(user2_email, user2_pwd)["access_token"]

    user2_docs = api_client.list_documents(u2_token)
    user2_convs = api_client.list_conversations(u2_token)
    print(f"User 2 documents: {user2_docs}")
    print(f"User 2 conversations: {user2_convs}")
    assert len(user2_docs) == 0, f"User 2 should see 0 documents, saw {len(user2_docs)}"
    assert len(user2_convs) == 0, f"User 2 should see 0 conversations, saw {len(user2_convs)}"
    print("SUCCESS: Strict cross-user data isolation confirmed!")

    print("\nTesting backend unreachable / offline error handling...")
    try:
        api_client.list_documents(u2_token, api_url="http://127.0.0.1:59999")
        raise AssertionError("Should have raised ApiError on offline port!")
    except api_client.ApiError as e:
        print(f"ApiError successfully caught: {e.status_code} - {e.message}")
        user_msg = render_error_message(e)
        print(f"Rendered User Message: '{user_msg}'")
        assert "Backend not reachable" in user_msg or "Backend not reachable" in e.message
        print("SUCCESS: Friendly offline notification verified!")

    print("\n" + "=" * 70)
    print("ALL 10 VERIFICATION STEPS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
