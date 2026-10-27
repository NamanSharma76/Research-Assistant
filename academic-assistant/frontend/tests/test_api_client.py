"""Unit tests for the frontend API client."""
import unittest
from unittest.mock import patch, MagicMock
import requests

from api_client import (
    ApiError,
    register_user,
    login_user,
    me,
    list_documents,
    upload_document,
    get_document,
    delete_document,
    get_document_sections,
    list_conversations,
    create_conversation,
    delete_conversation,
    list_messages,
    send_message,
    get_notes,
    get_outline,
    get_summary,
)


class TestApiClient(unittest.TestCase):
    def _make_mock_response(self, status_code: int, json_data: dict = None, headers: dict = None):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.ok = (status_code < 400)
        mock_resp.headers = headers or {}
        if json_data is not None:
            mock_resp.json.return_value = json_data
        else:
            mock_resp.json.side_effect = ValueError("No JSON")
        mock_resp.text = str(json_data) if json_data else ""
        return mock_resp

    @patch("requests.request")
    def test_register_user_success(self, mock_req):
        mock_req.return_value = self._make_mock_response(201, {"id": "u1", "email": "test@example.com"})
        res = register_user("test@example.com", "password123")
        self.assertEqual(res["id"], "u1")
        self.assertEqual(res["email"], "test@example.com")
        mock_req.assert_called_once()
        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs["json"], {"email": "test@example.com", "password": "password123"})

    @patch("requests.request")
    def test_login_user_success(self, mock_req):
        mock_req.return_value = self._make_mock_response(200, {"access_token": "tok123", "token_type": "bearer"})
        res = login_user("test@example.com", "password123")
        self.assertEqual(res["access_token"], "tok123")

    @patch("requests.request")
    def test_me(self, mock_req):
        mock_req.return_value = self._make_mock_response(200, {"id": "u1", "email": "test@example.com"})
        res = me("tok")
        self.assertEqual(res["id"], "u1")
        self.assertEqual(res["email"], "test@example.com")

    @patch("requests.request")
    def test_list_documents(self, mock_req):
        mock_req.return_value = self._make_mock_response(200, [{"id": "d1", "filename": "paper.pdf"}])
        res = list_documents("tok")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "d1")
        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer tok")

    @patch("requests.request")
    def test_upload_document(self, mock_req):
        mock_req.return_value = self._make_mock_response(202, {"id": "d1", "status": "processing"})
        res = upload_document("tok", filename="sample.pdf", file_bytes=b"%PDF-1.5 test")
        self.assertEqual(res["status"], "processing")
        _, kwargs = mock_req.call_args
        self.assertIn("file", kwargs["files"])
        self.assertEqual(kwargs["files"]["file"][0], "sample.pdf")

    @patch("requests.request")
    def test_get_document(self, mock_req):
        mock_req.return_value = self._make_mock_response(200, {"id": "d1", "status": "ready"})
        res = get_document("tok", "d1")
        self.assertEqual(res["status"], "ready")

    @patch("requests.request")
    def test_delete_document(self, mock_req):
        mock_req.return_value = self._make_mock_response(200, {"deleted": True})
        res = delete_document("tok", "d1")
        self.assertTrue(res)

    @patch("requests.request")
    def test_get_document_sections(self, mock_req):
        mock_req.return_value = self._make_mock_response(200, {"doc_id": "hash", "sections": ["Abstract", "Intro"]})
        res = get_document_sections("tok", "d1")
        self.assertEqual(res["sections"], ["Abstract", "Intro"])

    @patch("requests.request")
    def test_list_conversations(self, mock_req):
        mock_req.return_value = self._make_mock_response(200, [{"id": "c1", "document_id": "d1", "title": "New Chat"}])
        res = list_conversations("tok", "d1")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "c1")

    @patch("requests.request")
    def test_create_conversation(self, mock_req):
        mock_req.return_value = self._make_mock_response(201, {"id": "c1", "title": "Custom Title"})
        res = create_conversation("tok", "d1", title="Custom Title")
        self.assertEqual(res["id"], "c1")
        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs["json"], {"title": "Custom Title"})

    @patch("requests.request")
    def test_delete_conversation(self, mock_req):
        mock_req.return_value = self._make_mock_response(200, {"deleted": True})
        res = delete_conversation("tok", "c1")
        self.assertTrue(res)

    @patch("requests.request")
    def test_list_messages(self, mock_req):
        mock_req.return_value = self._make_mock_response(200, [{"id": "m1", "content": "Hello"}])
        res = list_messages("tok", "c1")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["content"], "Hello")

    @patch("requests.request")
    def test_send_message(self, mock_req):
        mock_req.return_value = self._make_mock_response(201, {
            "id": "m2",
            "role": "assistant",
            "content": "Answer",
            "intent": "paper_question",
            "trace": ["router", "answer"],
            "llm_calls": 2,
            "seconds": 1.2,
        })
        res = send_message("tok", "c1", "What is transformer?")
        self.assertEqual(res["role"], "assistant")
        self.assertEqual(res["llm_calls"], 2)

    @patch("requests.request")
    def test_get_notes(self, mock_req):
        mock_req.return_value = self._make_mock_response(200, {"problem": "NLP", "method": "Attention"})
        res = get_notes("tok", "d1", force=True)
        self.assertEqual(res["problem"], "NLP")
        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs["params"], {"force": "true"})

    @patch("requests.request")
    def test_get_outline(self, mock_req):
        mock_req.return_value = self._make_mock_response(200, {"title": "Deck", "slides": []})
        res = get_outline("tok", "d1", slides=7, force=False)
        self.assertEqual(res["title"], "Deck")
        _, kwargs = mock_req.call_args
        self.assertEqual(kwargs["params"], {"slides": 7})

    @patch("requests.request")
    def test_get_summary(self, mock_req):
        mock_req.return_value = self._make_mock_response(200, {"overall": "Summary text", "sections": {}})
        res = get_summary("tok", "d1")
        self.assertEqual(res["overall"], "Summary text")

    # Error handling tests
    @patch("requests.request")
    def test_error_401_unauthorized(self, mock_req):
        mock_req.return_value = self._make_mock_response(401, {"detail": "Invalid or expired token"})
        with self.assertRaises(ApiError) as ctx:
            list_documents("expired_token")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Session expired", ctx.exception.message)

    @patch("requests.request")
    def test_error_404_not_found(self, mock_req):
        mock_req.return_value = self._make_mock_response(404, {"detail": "Document not found"})
        with self.assertRaises(ApiError) as ctx:
            get_document("tok", "nonexistent")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("not found", ctx.exception.message)

    @patch("requests.request")
    def test_error_409_conflict(self, mock_req):
        mock_req.return_value = self._make_mock_response(409, {"detail": "Document is still processing"})
        with self.assertRaises(ApiError) as ctx:
            get_notes("tok", "d1")
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("Document is still processing", ctx.exception.message)

    @patch("requests.request")
    def test_error_429_rate_limit(self, mock_req):
        mock_req.return_value = self._make_mock_response(
            429,
            {"detail": "Rate limit exceeded. Try again in 20s."},
            headers={"Retry-After": "20"}
        )
        with self.assertRaises(ApiError) as ctx:
            send_message("tok", "c1", "Hello")
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.retry_after, 20)
        self.assertIn("Rate limit reached", ctx.exception.message)

    @patch("requests.request")
    def test_error_502_503_bad_gateway(self, mock_req):
        mock_req.return_value = self._make_mock_response(503, {"detail": "Upstream Groq error"})
        with self.assertRaises(ApiError) as ctx:
            send_message("tok", "c1", "Hello")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertFalse(ctx.exception.is_connection_error)
        self.assertIn("temporarily unavailable", ctx.exception.message)

    @patch("requests.request")
    def test_connection_error(self, mock_req):
        mock_req.side_effect = requests.exceptions.ConnectionError("Failed to connect")
        with self.assertRaises(ApiError) as ctx:
            list_documents("tok")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertTrue(ctx.exception.is_connection_error)
        self.assertEqual(ctx.exception.message, "Backend not reachable. Check that the server is running")

    @patch("requests.request")
    def test_timeout_error(self, mock_req):
        mock_req.side_effect = requests.exceptions.Timeout("Request timed out")
        with self.assertRaises(ApiError) as ctx:
            list_documents("tok")
        self.assertEqual(ctx.exception.status_code, 504)
        self.assertTrue(ctx.exception.is_connection_error)
        self.assertEqual(ctx.exception.message, "Backend not reachable. Check that the server is running")


if __name__ == "__main__":
    unittest.main()
