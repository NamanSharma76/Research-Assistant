"""FastAPI dependency providers for authentication and resource ownership."""
from typing import Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import User, Document, Conversation
from app.api.security import decode_token

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Validates the JWT token in Authorization: Bearer <token> and loads the current User."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deleted",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_owned_document(
    id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Document:
    """Loads a Document by id belonging strictly to the current user. Returns 404 if not found."""
    doc = db.query(Document).filter(Document.id == id, Document.user_id == user.id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return doc


def get_owned_conversation(
    id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Conversation:
    """Loads a Conversation by id belonging strictly to the current user. Returns 404 if not found."""
    conv = db.query(Conversation).filter(Conversation.id == id, Conversation.user_id == user.id).first()
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conv
