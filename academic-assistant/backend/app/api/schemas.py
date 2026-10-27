"""Pydantic schemas for request and response models across API endpoints."""
import re
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# Standard simple regex for email validation without extra packages
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ==========================================
# Auth Schemas
# ==========================================

class UserRegisterRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_REGEX.match(v):
            raise ValueError("Invalid email format")
        return v


class UserLoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_REGEX.match(v):
            raise ValueError("Invalid email format")
        return v


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ==========================================
# Document Schemas
# ==========================================

class DocumentResponse(BaseModel):
    id: str
    user_id: str
    doc_id: str
    filename: str
    status: str
    error: Optional[str] = None
    num_chunks: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentSectionsResponse(BaseModel):
    doc_id: str
    sections: List[str]


# ==========================================
# Conversation & Message Schemas
# ==========================================

class ConversationCreateRequest(BaseModel):
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    document_id: str
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    intent: Optional[str] = None
    trace: Optional[List[str]] = None
    llm_calls: Optional[int] = None
    seconds: Optional[float] = None
    kind: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}
