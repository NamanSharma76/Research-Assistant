"""Chat service managing turn locking, rate limiting, and DB message synchronization."""
import time
import threading
from collections import defaultdict
from typing import Set, Dict, Any, Tuple
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.config import CHAT_HISTORY_LOAD_TURNS
from app.db.models import Conversation, Message
from app.agents.graph import ChatSession
from app.llm import reset_call_count, get_call_count

# Per-conversation turn lock
_active_conversations: Set[str] = set()
_conv_lock = threading.Lock()

# Per-user sliding-window rate limiter: 20 messages per minute
_user_timestamps: Dict[str, list] = defaultdict(list)
_rate_lock = threading.Lock()
RATE_LIMIT_MAX_MESSAGES = 20
RATE_LIMIT_WINDOW_SECONDS = 60.0


def check_rate_limit(user_id: str) -> None:
    """Enforces 20 messages/minute sliding window per user. Raises 429 when exceeded."""
    now = time.time()
    with _rate_lock:
        window_start = now - RATE_LIMIT_WINDOW_SECONDS
        timestamps = [t for t in _user_timestamps[user_id] if t > window_start]
        if len(timestamps) >= RATE_LIMIT_MAX_MESSAGES:
            oldest_in_window = timestamps[0]
            retry_after = int(oldest_in_window + RATE_LIMIT_WINDOW_SECONDS - now) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded: maximum 20 messages per minute",
                headers={"Retry-After": str(max(1, retry_after))},
            )
        timestamps.append(now)
        _user_timestamps[user_id] = timestamps


def acquire_conversation_turn(conversation_id: str) -> bool:
    """Attempts to acquire the in-flight lock for a conversation. Returns False if already running."""
    with _conv_lock:
        if conversation_id in _active_conversations:
            return False
        _active_conversations.add(conversation_id)
        return True


def release_conversation_turn(conversation_id: str) -> None:
    """Releases the in-flight lock for a conversation."""
    with _conv_lock:
        _active_conversations.discard(conversation_id)


def execute_chat_turn(
    db: Session,
    conversation: Conversation,
    content: str,
) -> Message:
    """Builds ChatSession from DB history and executes turn.
    
    Returns saved assistant Message. Raises HTTPException on error.
    """
    doc_id = conversation.document.doc_id

    # 1. Load history from messages
    # Each turn has 1 user message and 1 assistant message, so CHAT_HISTORY_LOAD_TURNS turns = turns * 2 messages
    all_msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )

    history = []
    # Take the last CHAT_HISTORY_LOAD_TURNS turns (up to 2 * turns messages)
    history_slice = all_msgs[-(CHAT_HISTORY_LOAD_TURNS * 2):] if all_msgs else []
    for m in history_slice:
        history.append({"role": m.role, "content": m.content})

    # 2. Extract last_output from the most recent assistant message with kind
    last_output = {"kind": "none", "text": ""}
    for m in reversed(all_msgs):
        if m.role == "assistant" and m.kind:
            last_output = {"kind": m.kind, "text": m.content}
            break

    # 3. Save User message first
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=content,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # 4. Initialize and run ChatSession
    session = ChatSession(doc_id=doc_id, history=history, last_output=last_output)

    try:
        res = session.ask(content)
    except Exception as e:
        # LLM failure: return 502, do NOT save a broken assistant message
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM processing error: {str(e)}",
        )

    # 5. Determine output kind
    # Store kind on assistant message: paper_question, summarize, notes, outline, or inherited for rephrase
    kind = res.intent
    if res.intent == "rephrase_previous":
        kind = last_output.get("kind")
        if kind in ("none", None):
            kind = "rephrase"

    # 6. Save Assistant message
    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=res.answer,
        intent=res.intent,
        trace=res.trace,
        llm_calls=res.llm_calls,
        seconds=res.seconds,
        kind=kind,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return assistant_msg
