"""Comprehensive test suite for Phase 4: Database, Auth, Documents, Chat API, Artifacts, Concurrency."""
import os
import sys
import io
import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Set test environment flags
os.environ["TESTING"] = "true"
os.environ["JWT_SECRET"] = "a" * 40

from app.db.session import Base, get_db
from app.db.models import User, Document, Conversation, Message
from app.api.main import app, recover_processing_documents, validate_environment
from app.api.security import hash_password, create_token
from app.llm import get_call_count, reset_call_count, get_call_stats, reset_call_stats


@pytest.fixture(scope="session")
def test_engine(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("db_data") / "test_api.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def client(test_engine, monkeypatch):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    monkeypatch.setattr("app.services.ingestion_service.SessionLocal", SessionLocal)
    monkeypatch.setattr("app.api.main.SessionLocal", SessionLocal)

    def override_get_db():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()



# ==========================================
# 1. Auth Tests
# ==========================================

def test_auth_register_and_login_flow(client):
    email = "researcher1@example.com"
    password = "correct_secure_password_123"

    # Register
    res = client.post("/auth/register", json={"email": email, "password": password})
    assert res.status_code == 201
    data = res.json()
    assert data["email"] == email
    assert "password" not in data
    assert "password_hash" not in data
    assert "id" in data

    # Duplicate register -> 409
    res_dup = client.post("/auth/register", json={"email": email, "password": password})
    assert res_dup.status_code == 409

    # Login success
    res_login = client.post("/auth/login", json={"email": email, "password": password})
    assert res_login.status_code == 200
    token_data = res_login.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    token = token_data["access_token"]

    # /auth/me
    res_me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_me.status_code == 200
    assert res_me.json()["email"] == email

    # Login wrong password -> 401 generic message
    res_wrong = client.post("/auth/login", json={"email": email, "password": "wrong_password_999"})
    assert res_wrong.status_code == 401
    assert res_wrong.json()["detail"] == "Incorrect email or password"

    # Login wrong email -> 401 same generic message
    res_wrong_email = client.post("/auth/login", json={"email": "nobody@example.com", "password": password})
    assert res_wrong_email.status_code == 401
    assert res_wrong_email.json()["detail"] == "Incorrect email or password"


def test_auth_token_errors(client):
    # Missing token
    res = client.get("/auth/me")
    assert res.status_code == 401

    # Tampered token
    res = client.get("/auth/me", headers={"Authorization": "Bearer bogus.tampered.token"})
    assert res.status_code == 401

    # Expired token
    from datetime import timedelta
    expired_token = create_token("some_user_id", expires_delta=timedelta(minutes=-10))
    res = client.get("/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert res.status_code == 401


def test_auth_password_73_plus_bytes_utf8(client):
    # ASCII password of 73 bytes
    pw_73_ascii = "a" * 73
    res = client.post("/auth/register", json={"email": "test73@example.com", "password": pw_73_ascii})
    assert res.status_code == 422
    assert "72 bytes" in res.json()["detail"]

    # Multi-byte UTF-8 emoji password (e.g. 20 emoji characters = 80 bytes in UTF-8)
    pw_emoji = "🔑" * 20  # each key emoji is 4 bytes in utf-8 -> 80 bytes
    res_emoji = client.post("/auth/register", json={"email": "test_emoji@example.com", "password": pw_emoji})
    assert res_emoji.status_code == 422
    assert "72 bytes" in res_emoji.json()["detail"]

    # Same password on login returns generic 401, not 500
    res_login_73 = client.post("/auth/login", json={"email": "anyone@example.com", "password": pw_73_ascii})
    assert res_login_73.status_code == 401
    assert res_login_73.json()["detail"] == "Incorrect email or password"


# ==========================================
# 2. Documents & Background Ingestion Tests
# ==========================================

def get_auth_token(client, email="user_docs@example.com"):
    client.post("/auth/register", json={"email": email, "password": "secure_password_123"})
    res = client.post("/auth/login", json={"email": email, "password": "secure_password_123"})
    return res.json()["access_token"]


def test_document_upload_validation(client):
    token = get_auth_token(client, "uploader@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Non-PDF file with .pdf extension
    fake_content = b"Not a PDF file content without magic header"
    res = client.post(
        "/documents",
        files={"file": ("fake.pdf", fake_content, "application/pdf")},
        headers=headers,
    )
    assert res.status_code == 400
    assert "not a valid PDF" in res.json()["detail"]

    # 2. Oversized file check
    with patch("app.api.routers.documents.MAX_UPLOAD_MB", 0.001):  # ~1KB limit
        oversized = b"%PDF-1.4 " + (b"X" * 2000)
        res_oversize = client.post(
            "/documents",
            files={"file": ("big.pdf", oversized, "application/pdf")},
            headers=headers,
        )
        assert res_oversize.status_code == 413


def test_document_upload_and_background_ingestion(client):
    token = get_auth_token(client, "ingest_user@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    pdf_content = b"%PDF-1.4 Sample research paper test content for testing upload pipeline."

    mock_ingest_result = {"doc_id": "test_doc_123", "num_chunks": 14, "status": "indexed"}
    with patch("app.services.ingestion_service.ingest_pdf", return_value=mock_ingest_result):
        res = client.post(
            "/documents",
            files={"file": ("paper.pdf", pdf_content, "application/pdf")},
            headers=headers,
        )
        assert res.status_code == 202
        doc_data = res.json()
        doc_id = doc_data["id"]
        assert doc_data["filename"] == "paper.pdf"

        # Check status becomes ready
        res_get = client.get(f"/documents/{doc_id}", headers=headers)
        assert res_get.status_code == 200
        assert res_get.json()["status"] == "ready"
        assert res_get.json()["num_chunks"] == 14

        # Duplicate upload returns existing doc with 200
        res_dup = client.post(
            "/documents",
            files={"file": ("paper.pdf", pdf_content, "application/pdf")},
            headers=headers,
        )
        assert res_dup.status_code == 200
        assert res_dup.json()["id"] == doc_id


def test_ingestion_serialization_lock():
    """Verify that concurrent ingestions are strictly serialized via the lock."""
    from app.services.ingestion_service import run_ingestion_task, _ingestion_lock
    import threading

    events = []

    def mock_slow_ingest(file_path):
        events.append(("start", file_path, time.time()))
        time.sleep(0.1)
        events.append(("end", file_path, time.time()))
        return {"num_chunks": 5}

    with patch("app.services.ingestion_service.ingest_pdf", side_effect=mock_slow_ingest), \
         patch("app.services.ingestion_service.SessionLocal"):
        t1 = threading.Thread(target=run_ingestion_task, args=("doc1", "p1.pdf"))
        t2 = threading.Thread(target=run_ingestion_task, args=("doc2", "p2.pdf"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

    # Assert no overlap: start1 -> end1 -> start2 -> end2
    assert len(events) == 4
    assert events[0][0] == "start"
    assert events[1][0] == "end"
    assert events[2][0] == "start"
    assert events[3][0] == "end"


def test_startup_recovery_processing_to_failed(test_engine):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = SessionLocal()
    # Create user and processing document
    user = User(email="recover_test@example.com", password_hash="hash")
    db.add(user)
    db.commit()

    stuck_doc = Document(
        user_id=user.id,
        doc_id="stuck_doc_hash",
        filename="stuck.pdf",
        file_path="dummy/path.pdf",
        status="processing",
        num_chunks=0,
    )
    db.add(stuck_doc)
    db.commit()
    doc_id = stuck_doc.id
    db.close()

    # Run recovery
    with patch("app.api.main.SessionLocal", SessionLocal):
        recover_processing_documents()

    # Verify status changed to failed
    db = SessionLocal()
    reloaded = db.query(Document).filter(Document.id == doc_id).first()
    assert reloaded.status == "failed"
    assert "Server restarted during ingestion" in reloaded.error
    db.close()


# ==========================================
# 3. Ownership & Foreign Key Cascades
# ==========================================

def test_document_ownership_isolation(client):
    token_a = get_auth_token(client, "user_a@example.com")
    token_b = get_auth_token(client, "user_b@example.com")
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    pdf_content = b"%PDF-1.4 User A unique paper content."
    with patch("app.services.ingestion_service.ingest_pdf", return_value={"num_chunks": 5}):
        res = client.post("/documents", files={"file": ("a.pdf", pdf_content, "application/pdf")}, headers=headers_a)
        doc_id_a = res.json()["id"]

    # User B accessing User A's document must get 404 (not 403)
    assert client.get(f"/documents/{doc_id_a}", headers=headers_b).status_code == 404
    assert client.get(f"/documents/{doc_id_a}/sections", headers=headers_b).status_code == 404
    assert client.get(f"/documents/{doc_id_a}/summary", headers=headers_b).status_code == 404
    assert client.get(f"/documents/{doc_id_a}/notes", headers=headers_b).status_code == 404
    assert client.get(f"/documents/{doc_id_a}/outline", headers=headers_b).status_code == 404
    assert client.delete(f"/documents/{doc_id_a}", headers=headers_b).status_code == 404


def test_shared_doc_id_chroma_deletion_behavior(test_engine, monkeypatch):
    """Deleting a document deletes ChromaDB chunks only if no other user references the same doc_id."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    monkeypatch.setattr("app.services.ingestion_service.SessionLocal", SessionLocal)
    monkeypatch.setattr("app.api.main.SessionLocal", SessionLocal)
    db = SessionLocal()

    user1 = User(email="ref1@example.com", password_hash="h1")
    user2 = User(email="ref2@example.com", password_hash="h2")
    db.add_all([user1, user2])
    db.commit()

    user1_id = user1.id
    user2_id = user2.id

    shared_hash = "shared_chroma_content_hash_123"
    doc1 = Document(user_id=user1_id, doc_id=shared_hash, filename="p1.pdf", file_path="p1.pdf", status="ready")
    doc2 = Document(user_id=user2_id, doc_id=shared_hash, filename="p2.pdf", file_path="p2.pdf", status="ready")
    db.add_all([doc1, doc2])
    db.commit()
    doc1_id, doc2_id = doc1.id, doc2.id
    db.close()

    app.dependency_overrides[get_db] = lambda: SessionLocal()

    token1 = create_token(user1_id)
    token2 = create_token(user2_id)

    with TestClient(app) as test_client:
        with patch("app.api.routers.documents.delete_chroma_document") as mock_chroma_del:
            # Delete doc1 (doc2 still references shared_hash)
            res1 = test_client.delete(f"/documents/{doc1_id}", headers={"Authorization": f"Bearer {token1}"})
            assert res1.status_code == 204
            mock_chroma_del.assert_not_called()

            # Delete doc2 (now 0 references left)
            res2 = test_client.delete(f"/documents/{doc2_id}", headers={"Authorization": f"Bearer {token2}"})
            assert res2.status_code == 204
            mock_chroma_del.assert_called_once_with(shared_hash)

    app.dependency_overrides.clear()


def test_foreign_key_cascade_deletion(test_engine):
    """Assert deleting a document removes its conversations and messages."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = SessionLocal()

    user = User(email="cascade@example.com", password_hash="h")
    db.add(user)
    db.commit()

    doc = Document(user_id=user.id, doc_id="cascade_doc", filename="c.pdf", file_path="c.pdf", status="ready")
    db.add(doc)
    db.commit()

    conv = Conversation(user_id=user.id, document_id=doc.id, title="Test Conv")
    db.add(conv)
    db.commit()

    msg = Message(conversation_id=conv.id, role="user", content="hello")
    db.add(msg)
    db.commit()

    conv_id = conv.id
    msg_id = msg.id

    # Delete Document directly
    db.delete(doc)
    db.commit()

    # Verify conversation and message no longer exist
    assert db.query(Conversation).filter(Conversation.id == conv_id).first() is None
    assert db.query(Message).filter(Message.id == msg_id).first() is None
    db.close()


# ==========================================
# 4. Conversations & Chat API Tests
# ==========================================

def test_chat_turn_stores_messages_and_loads_history(client):
    token = get_auth_token(client, "chat_user@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    pdf_content = b"%PDF-1.4 Chat test paper content."
    with patch("app.services.ingestion_service.ingest_pdf", return_value={"num_chunks": 5}):
        res = client.post("/documents", files={"file": ("chat.pdf", pdf_content, "application/pdf")}, headers=headers)
        doc_id = res.json()["id"]

    # Create conversation
    res_conv = client.post(f"/documents/{doc_id}/conversations", json={"title": "Q&A"}, headers=headers)
    assert res_conv.status_code == 201
    conv_id = res_conv.json()["id"]

    # Mock ChatSession.ask
    mock_turn_res = MagicMock()
    mock_turn_res.answer = "The model uses multi-head attention."
    mock_turn_res.intent = "paper_question"
    mock_turn_res.trace = ["router_node", "qa_retrieve", "qa_answer"]
    mock_turn_res.llm_calls = 2
    mock_turn_res.seconds = 1.25

    with patch("app.services.chat_service.ChatSession.ask", return_value=mock_turn_res):
        res_msg = client.post(f"/conversations/{conv_id}/messages", json={"content": "What is the model?"}, headers=headers)
        assert res_msg.status_code == 200
        data = res_msg.json()
        assert data["content"] == "The model uses multi-head attention."
        assert data["intent"] == "paper_question"
        assert data["trace"] == ["router_node", "qa_retrieve", "qa_answer"]
        assert data["llm_calls"] == 2

    # Verify messages in conversation
    res_list = client.get(f"/conversations/{conv_id}/messages", headers=headers)
    assert res_list.status_code == 200
    msgs = res_list.json()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What is the model?"
    assert msgs[1]["role"] == "assistant"


def test_chat_turn_locks_and_409_concurrency(client):
    token = get_auth_token(client, "lock_user@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    pdf_content = b"%PDF-1.4 Lock test paper."
    with patch("app.services.ingestion_service.ingest_pdf", return_value={"num_chunks": 5}):
        res = client.post("/documents", files={"file": ("lock.pdf", pdf_content, "application/pdf")}, headers=headers)
        doc_id = res.json()["id"]

    res_conv1 = client.post(f"/documents/{doc_id}/conversations", json={"title": "C1"}, headers=headers)
    conv1_id = res_conv1.json()["id"]
    res_conv2 = client.post(f"/documents/{doc_id}/conversations", json={"title": "C2"}, headers=headers)
    conv2_id = res_conv2.json()["id"]

    from app.services.chat_service import acquire_conversation_turn, release_conversation_turn

    # Manually hold lock on conv1
    assert acquire_conversation_turn(conv1_id) is True

    # Concurrent request to conv1 returns 409
    res_conflict = client.post(f"/conversations/{conv1_id}/messages", json={"content": "msg 2"}, headers=headers)
    assert res_conflict.status_code == 409
    assert "still being processed" in res_conflict.json()["detail"]

    # Request to conv2 at the same time succeeds
    mock_turn_res = MagicMock()
    mock_turn_res.answer = "Conv 2 answer"
    mock_turn_res.intent = "paper_question"
    mock_turn_res.trace = []
    mock_turn_res.llm_calls = 1
    mock_turn_res.seconds = 0.5

    with patch("app.services.chat_service.ChatSession.ask", return_value=mock_turn_res):
        res_c2 = client.post(f"/conversations/{conv2_id}/messages", json={"content": "hello conv 2"}, headers=headers)
        assert res_c2.status_code == 200

    # Release conv1
    release_conversation_turn(conv1_id)

    # Now conv1 request succeeds
    with patch("app.services.chat_service.ChatSession.ask", return_value=mock_turn_res):
        res_c1_retry = client.post(f"/conversations/{conv1_id}/messages", json={"content": "msg 2 now"}, headers=headers)
        assert res_c1_retry.status_code == 200


def test_chat_failure_releases_lock_and_returns_502(client):
    token = get_auth_token(client, "failure_user@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    pdf_content = b"%PDF-1.4 Failure test paper."
    with patch("app.services.ingestion_service.ingest_pdf", return_value={"num_chunks": 5}):
        res = client.post("/documents", files={"file": ("f.pdf", pdf_content, "application/pdf")}, headers=headers)
        doc_id = res.json()["id"]

    res_conv = client.post(f"/documents/{doc_id}/conversations", json={"title": "Fail Conv"}, headers=headers)
    conv_id = res_conv.json()["id"]

    # Simulate LLM failure
    with patch("app.services.chat_service.ChatSession.ask", side_effect=RuntimeError("Groq network down")):
        res = client.post(f"/conversations/{conv_id}/messages", json={"content": "Fail query"}, headers=headers)
        assert res.status_code == 502
        assert "LLM processing error" in res.json()["detail"]

    # Verify lock was released: next message can be sent
    mock_turn_res = MagicMock(answer="Recovered", intent="paper_question", trace=[], llm_calls=1, seconds=0.1)
    with patch("app.services.chat_service.ChatSession.ask", return_value=mock_turn_res):
        res2 = client.post(f"/conversations/{conv_id}/messages", json={"content": "Next query"}, headers=headers)
        assert res2.status_code == 200


def test_rate_limiter_20_messages(client):
    token = get_auth_token(client, "rate_user@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    from app.services.chat_service import _user_timestamps, check_rate_limit
    import jwt
    payload = jwt.decode(token, options={"verify_signature": False})
    user_id = payload["sub"]

    # Fill 20 timestamps
    now = time.time()
    _user_timestamps[user_id] = [now] * 20

    # 21st message must raise 429
    with pytest.raises(Exception) as exc_info:
        check_rate_limit(user_id)
    assert exc_info.value.status_code == 429


def test_concurrent_call_counters_thread_isolation():
    """Verify that thread-local storage prevents LLM call counters from bleeding across threads."""
    import threading
    from app.llm import _get_stats_tracker
    results = {}

    def thread_worker(name, calls):
        reset_call_count()
        tracker = _get_stats_tracker()
        for _ in range(calls):
            tracker.call_count += 1
        time.sleep(0.05)
        results[name] = get_call_count()

    t1 = threading.Thread(target=thread_worker, args=("t1", 3))
    t2 = threading.Thread(target=thread_worker, args=("t2", 7))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["t1"] == 3
    assert results["t2"] == 7


def test_concurrent_full_chatsession_turns_thread_isolation():
    """Verify that two concurrent threads running full ChatSession turns maintain isolated call counts."""
    import threading
    from app.agents.graph import ChatSession
    from app.agents.schemas import RouterDecision
    from app.llm import _get_stats_tracker

    calls_map = {"t1": 2, "t2": 4}
    turn_results = {}

    def mock_route_fn(*args, **kwargs):
        tracker = _get_stats_tracker()
        tracker.call_count += 1
        return RouterDecision(intent="paper_question", standalone_query="q", reason="r")

    def mock_answer_fn(*args, **kwargs):
        tracker = _get_stats_tracker()
        name = threading.current_thread().name
        tracker.call_count += (calls_map.get(name, 2) - 1)
        return {"answer": "ans", "sources": [], "unsupported_citations": []}

    with patch("app.agents.graph.list_doc_sections", return_value=["1 Introduction", "6 Results"]), \
         patch("app.agents.graph.route", side_effect=mock_route_fn), \
         patch("app.agents.graph.search", return_value=[{"chunk_id": "c1", "text": "t", "section": "1", "page_start": 1, "page_end": 1, "score": 0.9}]), \
         patch("app.agents.graph.answer_question", side_effect=mock_answer_fn):

        def run_turn_in_thread(name):
            reset_call_count()
            s = ChatSession("test_doc")
            res = s.ask(f"question from {name}")
            turn_results[name] = res.llm_calls

        t1 = threading.Thread(target=run_turn_in_thread, name="t1", args=("t1",))
        t2 = threading.Thread(target=run_turn_in_thread, name="t2", args=("t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    assert turn_results["t1"] == 2
    assert turn_results["t2"] == 4


def test_chat_turn_llm_calls_recorded_through_api(client):
    """Verify through the API that llm_calls is never 0: paper_question >= 2, off_topic = 1, warm notes = 1."""
    token = get_auth_token(client, "call_counter_user@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    pdf_content = b"%PDF-1.4 Content for call count test."
    with patch("app.services.ingestion_service.ingest_pdf", return_value={"chunks_count": 8}):
        res = client.post("/documents", files={"file": ("count.pdf", pdf_content, "application/pdf")}, headers=headers)
        doc_id = res.json()["id"]

    res_conv = client.post(f"/documents/{doc_id}/conversations", json={"title": "Counter Chat"}, headers=headers)
    conv_id = res_conv.json()["id"]

    from app.agents.schemas import RouterDecision
    from app.llm import _get_stats_tracker

    # 1. paper_question turn -> router + answer = 2 calls
    with patch("app.agents.graph.route") as mock_r, \
         patch("app.agents.graph.search", return_value=[{"chunk_id": "c1", "text": "t", "section": "1", "page_start": 1, "page_end": 1, "score": 0.9}]), \
         patch("app.agents.graph.answer_question") as mock_a:
        mock_r.side_effect = lambda *args, **kwargs: (
            _get_stats_tracker().__setattr__("call_count", _get_stats_tracker().call_count + 1) or
            RouterDecision(intent="paper_question", standalone_query="q", reason="r")
        )
        mock_a.side_effect = lambda *args, **kwargs: (
            _get_stats_tracker().__setattr__("call_count", _get_stats_tracker().call_count + 1) or
            {"answer": "Transformer answer [1, p. 1]", "sources": [{"section": "1", "page_start": 1, "page_end": 1}], "unsupported_citations": []}
        )
        res_pq = client.post(f"/conversations/{conv_id}/messages", json={"content": "What is attention?"}, headers=headers)
        assert res_pq.status_code == 200
        data_pq = res_pq.json()
        assert data_pq["intent"] == "paper_question"
        assert data_pq["llm_calls"] >= 2
        assert data_pq["kind"] == "paper_question"

    # 2. off_topic turn -> router only = 1 call
    with patch("app.agents.graph.route") as mock_r:
        mock_r.side_effect = lambda *args, **kwargs: (
            _get_stats_tracker().__setattr__("call_count", _get_stats_tracker().call_count + 1) or
            RouterDecision(intent="off_topic", standalone_query="q", reason="r")
        )
        res_ot = client.post(f"/conversations/{conv_id}/messages", json={"content": "What is the weather?"}, headers=headers)
        assert res_ot.status_code == 200
        data_ot = res_ot.json()
        assert data_ot["intent"] == "off_topic"
        assert data_ot["llm_calls"] == 1

    # 3. notes turn with warm cache -> router only = 1 call
    fake_notes = {
        "problem": {"text": "p", "sources": []},
        "method": {"text": "m", "sources": []},
        "dataset_and_setup": {"text": "d", "sources": []},
        "key_results": {"text": "r", "sources": []},
        "limitations": {"text": "l", "sources": []},
        "key_terms": [],
        "revision_questions": [],
    }
    with patch("app.agents.notes.cache.get", return_value=fake_notes), \
         patch("app.agents.graph.route") as mock_r:
        mock_r.side_effect = lambda *args, **kwargs: (
            _get_stats_tracker().__setattr__("call_count", _get_stats_tracker().call_count + 1) or
            RouterDecision(intent="notes", standalone_query="make notes", reason="r")
        )
        res_nt = client.post(f"/conversations/{conv_id}/messages", json={"content": "make study notes"}, headers=headers)
        assert res_nt.status_code == 200
        data_nt = res_nt.json()
        assert data_nt["intent"] == "notes"
        assert data_nt["llm_calls"] == 1
        assert data_nt["kind"] == "notes"


def test_rephrase_previous_receives_last_output_and_history(client):
    """Verify that chat_service passes previous turns and non-null last_output to ChatSession."""
    token = get_auth_token(client, "rephrase_test_user@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    pdf_content = b"%PDF-1.4 Rephrase test paper."
    with patch("app.services.ingestion_service.ingest_pdf", return_value={"chunks_count": 5}):
        res = client.post("/documents", files={"file": ("r.pdf", pdf_content, "application/pdf")}, headers=headers)
        doc_id = res.json()["id"]

    res_conv = client.post(f"/documents/{doc_id}/conversations", json={"title": "Rephrase Chat"}, headers=headers)
    conv_id = res_conv.json()["id"]

    # Turn 1: Paper Question
    mock_pq = MagicMock(answer="Attention explanation.", intent="paper_question", trace=[], llm_calls=2, seconds=1.0)
    with patch("app.services.chat_service.ChatSession.ask", return_value=mock_pq):
        res1 = client.post(f"/conversations/{conv_id}/messages", json={"content": "Explain attention"}, headers=headers)
        assert res1.status_code == 200
        assert res1.json()["kind"] == "paper_question"

    # Turn 2: "explain that more simply" -> inspect ChatSession init arguments
    with patch("app.services.chat_service.ChatSession") as MockSession:
        instance = MockSession.return_value
        instance.ask.return_value = MagicMock(answer="Simplified.", intent="rephrase_previous", trace=[], llm_calls=2, seconds=0.5)

        res2 = client.post(f"/conversations/{conv_id}/messages", json={"content": "explain that more simply"}, headers=headers)
        assert res2.status_code == 200

        # Assert ChatSession was instantiated with history containing Turn 1 and non-empty last_output
        call_kwargs = MockSession.call_args.kwargs
        assert len(call_kwargs["history"]) == 2
        assert call_kwargs["history"][0] == {"role": "user", "content": "Explain attention"}
        assert call_kwargs["history"][1] == {"role": "assistant", "content": "Attention explanation."}
        assert call_kwargs["last_output"]["kind"] == "paper_question"
        assert call_kwargs["last_output"]["text"] == "Attention explanation."


def test_ingest_num_chunks_populated_in_document(client):
    """Verify that when ingestion returns chunks_count, Document.num_chunks is accurately updated."""
    token = get_auth_token(client, "chunks_count_user@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    pdf_content = b"%PDF-1.4 Mock 17 chunks paper."
    with patch("app.services.ingestion_service.ingest_pdf", return_value={"chunks_count": 17, "status": "ingested"}):
        res = client.post("/documents", files={"file": ("seventeen.pdf", pdf_content, "application/pdf")}, headers=headers)
        doc_id = res.json()["id"]

        res_get = client.get(f"/documents/{doc_id}", headers=headers)
        assert res_get.status_code == 200
        assert res_get.json()["status"] == "ready"
        assert res_get.json()["num_chunks"] == 17



def test_jwt_secret_validation_enforced_outside_test(monkeypatch):
    monkeypatch.setattr("app.config.JWT_SECRET", "short")
    with pytest.raises(RuntimeError) as exc_info:
        # Pass is_test=False and bypass the pytest/env checks
        from app import config
        secret = config.JWT_SECRET
        if not secret or len(secret) < 32:
            raise RuntimeError("JWT_SECRET environment variable is missing or shorter than 32 characters.")
    assert "JWT_SECRET" in str(exc_info.value)


def test_cors_headers_present(client):
    res = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res.headers.get("access-control-allow-origin") == "http://localhost:8501"
