"""HTTP API Client for Academic Research & Learning Assistant.

Communicates exclusively with the FastAPI backend over HTTP using requests.
No direct database, agent, ChromaDB, or internal backend imports.
"""
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import requests

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_STANDARD = 15.0
TIMEOUT_LONG = 300.0


class ApiError(Exception):
    """Normalized API error for all backend HTTP interactions."""

    def __init__(
        self,
        status_code: int,
        message: str,
        detail: Any = None,
        retry_after: Optional[int] = None,
        is_connection_error: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.detail = detail
        self.retry_after = retry_after
        self.is_connection_error = is_connection_error

    def __str__(self) -> str:
        return self.message


def _parse_retry_after(response: requests.Response, detail: str) -> Optional[int]:
    """Extracts retry-after duration in seconds from header or detail text."""
    header_val = response.headers.get("Retry-After")
    if header_val and header_val.isdigit():
        return int(header_val)
    match = re.search(r"try again in (\d+)", str(detail))
    if match:
        return int(match.group(1))
    return None


def _handle_response_error(response: requests.Response) -> ApiError:
    """Extracts backend detail and maps HTTP status code to ApiError."""
    status = response.status_code
    detail: Any = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = body.get("detail", body)
        else:
            detail = body
    except Exception:
        detail = response.text or ""

    retry_after = _parse_retry_after(response, str(detail))

    if status == 401:
        msg = "Session expired or invalid credentials. Please log in again."
    elif status == 404:
        msg = f"Resource not found: {detail}" if detail else "Requested resource was not found."
    elif status == 409:
        msg = str(detail) if detail else "Conflict: Document is still processing or turn is active."
    elif status == 422:
        msg = f"Validation error: {detail}"
    elif status == 429:
        if retry_after:
            msg = f"Rate limit reached. Please retry in {retry_after} seconds."
        else:
            msg = f"Rate limit reached. {detail}"
    elif status in (502, 503):
        msg = "Service temporarily unavailable or upstream AI model error. Please retry shortly."
    elif status >= 500:
        msg = f"Backend internal server error ({status})."
    else:
        msg = f"HTTP Error {status}: {detail}"

    return ApiError(status_code=status, message=msg, detail=detail, retry_after=retry_after)


def _request(
    method: str,
    path: str,
    token: Optional[str] = None,
    api_url: Optional[str] = None,
    **kwargs,
) -> requests.Response:
    """Centralized request dispatcher with timeouts, auth headers, and error normalization."""
    base = (api_url or os.getenv("API_URL", DEFAULT_API_URL)).rstrip("/")
    url = f"{base}{path}"

    headers = kwargs.pop("headers", {})
    if token:
        headers["Authorization"] = f"Bearer {token}"

    timeout = kwargs.pop("timeout", TIMEOUT_STANDARD)

    try:
        resp = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
    except requests.exceptions.Timeout:
        raise ApiError(
            status_code=504,
            message="Backend not reachable. Check that the server is running",
            is_connection_error=True,
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.RequestException):
        raise ApiError(
            status_code=503,
            message="Backend not reachable. Check that the server is running",
            is_connection_error=True,
        )

    if not resp.ok:
        raise _handle_response_error(resp)

    return resp


# ==========================================
# Authentication Endpoints
# ==========================================

def register(email: str, password: str, api_url: Optional[str] = None) -> Dict[str, Any]:
    """Registers a new user account."""
    resp = _request(
        "POST",
        "/auth/register",
        api_url=api_url,
        json={"email": email, "password": password},
        timeout=TIMEOUT_STANDARD,
    )
    return resp.json()


def login(email: str, password: str, api_url: Optional[str] = None) -> Dict[str, Any]:
    """Authenticates user and returns JWT access token."""
    resp = _request(
        "POST",
        "/auth/login",
        api_url=api_url,
        json={"email": email, "password": password},
        timeout=TIMEOUT_STANDARD,
    )
    return resp.json()


def me(token: str, api_url: Optional[str] = None) -> Dict[str, Any]:
    """Returns currently authenticated user profile."""
    resp = _request("GET", "/auth/me", token=token, api_url=api_url, timeout=TIMEOUT_STANDARD)
    return resp.json()


# ==========================================
# Document Endpoints
# ==========================================

def list_documents(token: str, api_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists all documents owned by the authenticated user."""
    resp = _request("GET", "/documents", token=token, api_url=api_url, timeout=TIMEOUT_STANDARD)
    return resp.json()


def get_document(token: str, doc_id: str, api_url: Optional[str] = None) -> Dict[str, Any]:
    """Gets details and status for a single document."""
    resp = _request("GET", f"/documents/{doc_id}", token=token, api_url=api_url, timeout=TIMEOUT_STANDARD)
    return resp.json()


def upload_document(
    token: str,
    file_name: Optional[str] = None,
    file_bytes: bytes = b"",
    api_url: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Uploads a PDF document for background ingestion."""
    actual_name = filename or file_name or "document.pdf"
    files = {"file": (actual_name, file_bytes, "application/pdf")}
    resp = _request(
        "POST",
        "/documents",
        token=token,
        api_url=api_url,
        files=files,
        timeout=TIMEOUT_STANDARD,
    )
    return resp.json()


def delete_document(token: str, doc_id: str, api_url: Optional[str] = None) -> bool:
    """Deletes a document and its associated conversations/messages."""
    resp = _request("DELETE", f"/documents/{doc_id}", token=token, api_url=api_url, timeout=TIMEOUT_STANDARD)
    return resp.status_code in (200, 204)


def list_sections(token: str, doc_id: str, api_url: Optional[str] = None) -> Dict[str, Any]:
    """Returns extracted section headers for a document."""
    resp = _request("GET", f"/documents/{doc_id}/sections", token=token, api_url=api_url, timeout=TIMEOUT_STANDARD)
    return resp.json()


# ==========================================
# Conversation & Message Endpoints
# ==========================================

def create_conversation(
    token: str,
    doc_id: str,
    title: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Creates a new conversation session for a document."""
    payload: Dict[str, Any] = {}
    if title:
        payload["title"] = title
    resp = _request(
        "POST",
        f"/documents/{doc_id}/conversations",
        token=token,
        api_url=api_url,
        json=payload,
        timeout=TIMEOUT_STANDARD,
    )
    return resp.json()


def list_conversations(
    token: str,
    doc_id: Optional[str] = None,
    api_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists conversations owned by the user, optionally filtered by document ID."""
    resp = _request("GET", "/conversations", token=token, api_url=api_url, timeout=TIMEOUT_STANDARD)
    convs = resp.json()
    if doc_id:
        return [c for c in convs if c.get("document_id") == doc_id]
    return convs


def delete_conversation(token: str, conv_id: str, api_url: Optional[str] = None) -> bool:
    """Deletes a conversation session and all its messages."""
    resp = _request("DELETE", f"/conversations/{conv_id}", token=token, api_url=api_url, timeout=TIMEOUT_STANDARD)
    return resp.status_code in (200, 204)


def list_messages(token: str, conv_id: str, api_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves chronological message history for a conversation."""
    resp = _request("GET", f"/conversations/{conv_id}/messages", token=token, api_url=api_url, timeout=TIMEOUT_STANDARD)
    return resp.json()


def send_message(
    token: str,
    conv_id: str,
    content: str,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Sends a user message to the conversation and returns assistant turn result."""
    resp = _request(
        "POST",
        f"/conversations/{conv_id}/messages",
        token=token,
        api_url=api_url,
        json={"content": content},
        timeout=TIMEOUT_LONG,
    )
    return resp.json()


# ==========================================
# Artifact Endpoints (Notes, Outline, Summary)
# ==========================================

def get_notes(
    token: str,
    doc_id: str,
    force: bool = False,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates or retrieves comprehensive structured study notes."""
    params = {"force": str(force).lower()} if force else {}
    resp = _request(
        "GET",
        f"/documents/{doc_id}/notes",
        token=token,
        api_url=api_url,
        params=params,
        timeout=TIMEOUT_LONG,
    )
    return resp.json()


def get_outline(
    token: str,
    doc_id: str,
    slides: int = 10,
    force: bool = False,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates or retrieves presentation slides outline."""
    params = {"slides": slides}
    if force:
        params["force"] = "true"
    resp = _request(
        "GET",
        f"/documents/{doc_id}/outline",
        token=token,
        api_url=api_url,
        params=params,
        timeout=TIMEOUT_LONG,
    )
    return resp.json()


def get_summary(
    token: str,
    doc_id: str,
    force: bool = False,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates or retrieves overall and section summaries."""
    params = {"force": str(force).lower()} if force else {}
    resp = _request(
        "GET",
        f"/documents/{doc_id}/summary",
        token=token,
        api_url=api_url,
        params=params,
        timeout=TIMEOUT_LONG,
    )
    return resp.json()


# Function aliases for convenient invocation
register_user = register
login_user = login
get_document_sections = list_sections
