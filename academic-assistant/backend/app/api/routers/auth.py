"""Authentication routes for registration, login, and current user info."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import User
from app.api.schemas import UserRegisterRequest, UserLoginRequest, UserResponse, TokenResponse
from app.api.security import hash_password, verify_password, create_token
from app.api.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(req: UserRegisterRequest, db: Session = Depends(get_db)):
    # Validate password length in bytes (8 to 72 bytes)
    pw_bytes = req.password.encode("utf-8")
    if len(pw_bytes) < 8 or len(pw_bytes) > 72:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be between 8 and 72 bytes in UTF-8",
        )

    norm_email = req.email.strip().lower()

    # Check for existing user
    existing = db.query(User).filter(User.email == norm_email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email is already registered",
        )

    hashed = hash_password(req.password)
    user = User(email=norm_email, password_hash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(req: UserLoginRequest, db: Session = Depends(get_db)):
    pw_bytes = req.password.encode("utf-8")
    # If password exceeds 72 bytes, return generic 401 instead of 500 error from bcrypt
    if len(pw_bytes) > 72:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    norm_email = req.email.strip().lower()
    user = db.query(User).filter(User.email == norm_email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_token(user.id)
    return {"access_token": token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user)):
    return user
