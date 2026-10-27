"""Conversations and message routing endpoints."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import User, Document, Conversation, Message
from app.api.schemas import (
    ConversationCreateRequest,
    ConversationResponse,
    MessageCreateRequest,
    MessageResponse,
)
from app.api.deps import (
    get_current_user,
    get_owned_document,
    get_owned_conversation,
)
from app.services.chat_service import (
    check_rate_limit,
    acquire_conversation_turn,
    release_conversation_turn,
    execute_chat_turn,
)

router = APIRouter(tags=["conversations"])


@router.post("/documents/{id}/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    req: ConversationCreateRequest,
    doc: Document = Depends(get_owned_document),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    title = req.title.strip() if req.title and req.title.strip() else f"Chat about {doc.filename}"
    conv = Conversation(
        user_id=user.id,
        document_id=doc.id,
        title=title,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@router.get("/conversations", response_model=List[ConversationResponse])
def list_conversations(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )
    return convs


@router.get("/conversations/{id}/messages", response_model=List[MessageResponse])
def get_conversation_messages(
    conv: Conversation = Depends(get_owned_conversation),
    db: Session = Depends(get_db),
):
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )
    return messages


@router.delete("/conversations/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conv: Conversation = Depends(get_owned_conversation),
    db: Session = Depends(get_db),
):
    db.delete(conv)
    db.commit()
    return None


@router.post("/conversations/{id}/messages", response_model=MessageResponse)
def post_conversation_message(
    req: MessageCreateRequest,
    conv: Conversation = Depends(get_owned_conversation),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 1. Document readiness check
    if conv.document.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document is in status '{conv.document.status}'; chat is not available until ready",
        )

    # 2. Rate limit check
    check_rate_limit(user.id)

    # 3. Acquire single-turn lock on conversation
    acquired = acquire_conversation_turn(conv.id)
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A previous message is still being processed",
        )

    try:
        assistant_msg = execute_chat_turn(db, conv, req.content.strip())
        return assistant_msg
    finally:
        release_conversation_turn(conv.id)
