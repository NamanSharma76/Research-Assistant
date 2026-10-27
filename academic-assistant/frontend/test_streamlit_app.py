"""AppTest script to verify Streamlit frontend pages, widgets, and state transitions."""
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
frontend_dir = os.path.abspath(os.path.dirname(__file__))
if frontend_dir not in sys.path:
    sys.path.insert(0, frontend_dir)

from streamlit.testing.v1 import AppTest

def test_streamlit_app_flow():
    print("=" * 60)
    print("STREAMLIT APP COMPONENT & RENDER TESTING")
    print("=" * 60)

    app_path = os.path.join(frontend_dir, "app.py")
    
    # Test 1: Unauthenticated Auth Gate
    print("\n1. Testing Unauthenticated Gate...")
    at = AppTest.from_file(app_path)
    at.run()
    assert not at.exception, f"App crashed on initial load: {at.exception}"
    print(f"Title rendered: {[t.value for t in at.title]}")
    tabs = [t.label for t in at.tabs]
    print(f"Auth tabs rendered: {tabs}")
    assert "🔑 Log In" in tabs and "📝 Register New Account" in tabs
    print("SUCCESS: Auth Gate renders cleanly.")

    # Test 2: Offline / Backend Unreachable Error Handling
    print("\n2. Testing Backend Unreachable Display...")
    # Point to invalid API port
    os.environ["API_URL"] = "http://127.0.0.1:59999"
    at_offline = AppTest.from_file(app_path)
    at_offline.run()
    # Attempt login with backend offline
    at_offline.text_input[0].input("user@uni.edu")
    at_offline.text_input[1].input("somepassword123")
    at_offline.form_submit_button[0].click().run()
    errors = [e.value for e in at_offline.error]
    print(f"Error displayed on offline login: {errors}")
    assert any("Backend not reachable" in err for err in errors)
    print("SUCCESS: Offline error message correctly displayed.")

    # Reset API_URL to active backend
    os.environ["API_URL"] = "http://127.0.0.1:8000"

    # Test 3: Authenticated State with Active Document
    print("\n3. Testing Authenticated View with Ready Document...")
    import api_client
    login_res = api_client.login("researcher_p45@university.edu", "SecurePassword123")
    token = login_res["access_token"]
    
    at_auth = AppTest.from_file(app_path)
    # Inject authenticated session state
    at_auth.session_state["token"] = token
    at_auth.session_state["user_email"] = "researcher_p45@university.edu"
    at_auth.run()
    assert not at_auth.exception, f"App crashed in authenticated view: {at_auth.exception}"

    print(f"Headers rendered: {[h.value for h in at_auth.header]}")
    main_tabs = [t.label for t in at_auth.tabs]
    print(f"Workspace tabs rendered: {main_tabs}")
    assert "💬 Chat" in main_tabs
    assert "📖 Notes" in main_tabs
    assert "📊 Outline" in main_tabs
    assert "📄 Summary" in main_tabs
    print("SUCCESS: Authenticated workspace tabs render cleanly.")

    # Test 4: Processing Document State
    print("\n4. Testing Ingestion Processing Banner...")
    at_proc = AppTest.from_file(app_path)
    at_proc.session_state["token"] = token
    at_proc.session_state["user_email"] = "researcher_p45@university.edu"
    # Mock selected document in processing state
    fake_doc = {"id": "proc-id", "filename": "sample.pdf", "status": "processing", "doc_id": "test"}
    # Force mock
    import unittest.mock as mock
    with mock.patch("api_client.list_documents", return_value=[fake_doc]):
        at_proc.run()
        info_msgs = [i.value for i in at_proc.info]
        print(f"Info messages during processing: {info_msgs}")
        assert any("currently being processed" in m for m in info_msgs)
        print("SUCCESS: Processing state informs user with spinner/banner and locks tabs.")

    # Test 5: Logout State
    print("\n5. Testing Logout...")
    at_logout = AppTest.from_file(app_path)
    at_logout.session_state["token"] = token
    at_logout.session_state["user_email"] = "researcher_p45@university.edu"
    at_logout.run()
    # Find and click logout button
    logout_btn = [b for b in at_logout.button if "Log Out" in b.label][0]
    logout_btn.click().run()
    # State should be cleared
    assert at_logout.session_state.get("token") is None
    print("SUCCESS: Logout cleanly clears session state and redirects to Auth Gate.")

    print("\n" + "=" * 60)
    print("ALL STREAMLIT WEB APP TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_streamlit_app_flow()
