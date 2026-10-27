"""Password hashing and verification with bcrypt, and JWT token operations."""
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import bcrypt
import jwt
from fastapi import HTTPException, status
from app.config import JWT_SECRET, JWT_EXPIRE_MINUTES


def hash_password(password: str) -> str:
    """Hashes a password using bcrypt directly.
    
    Passwords longer than 72 bytes in UTF-8 must be rejected by callers before hashing.
    """
    pw_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pw_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Verifies a plain password against a bcrypt hash."""
    try:
        pw_bytes = plain_password.encode("utf-8")
        hash_bytes = password_hash.encode("utf-8")
        return bcrypt.checkpw(pw_bytes, hash_bytes)
    except Exception:
        return False


def _get_jwt_secret() -> str:
    from app import config
    return config.JWT_SECRET or "a" * 32


def create_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a signed HS256 JWT access token."""
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload: Dict[str, Any] = {
        "sub": user_id,
        "exp": expire,
        "iat": now,
    }
    secret = _get_jwt_secret()
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token: str) -> Dict[str, Any]:
    """Decodes and verifies a JWT token. Raises HTTPException(401) on failure."""
    try:
        secret = _get_jwt_secret()
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
