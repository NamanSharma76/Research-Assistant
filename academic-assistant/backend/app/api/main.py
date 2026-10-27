"""FastAPI application entry point, middleware, lifecycle events, and router registration."""
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from app.config import (
    CORS_ORIGINS,
    JWT_SECRET,
)
from app.db.session import engine, Base, SessionLocal
from app.db.models import Document
from app.api.routers import auth, documents, conversations, artifacts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")


def recover_processing_documents():
    """Startup recovery: mark stuck 'processing' documents as 'failed'."""
    db = SessionLocal()
    try:
        stuck_docs = db.query(Document).filter(Document.status == "processing").all()
        for doc in stuck_docs:
            doc.status = "failed"
            doc.error = "Server restarted during ingestion"
        if stuck_docs:
            db.commit()
            logger.info(f"Startup recovery: marked {len(stuck_docs)} interrupted document(s) as failed")
    except Exception as e:
        logger.error(f"Error recovering processing documents on startup: {e}")
    finally:
        db.close()


def validate_environment(is_test: bool = False):
    """Enforces JWT_SECRET presence and minimum length outside test environments."""
    if is_test or os.getenv("TESTING", "").lower() in ("true", "1") or "pytest" in sys.modules:
        return
    # Check current runtime config JWT_SECRET
    from app import config
    secret = config.JWT_SECRET
    if not secret or len(secret) < 32:
        raise RuntimeError(
            "JWT_SECRET environment variable is missing or shorter than 32 characters. "
            "Set a secure JWT_SECRET (32+ chars) in your .env before starting the server."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: validate env, create tables, run recovery
    validate_environment()
    Base.metadata.create_all(bind=engine)
    recover_processing_documents()
    yield
    # Shutdown logic (if any)


app = FastAPI(
    title="Academic Research & Learning Assistant API",
    version="1.0.0",
    description="HTTP API for Academic RAG Assistant with authentication, document management, chat, and agentic artifacts.",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.streamlit\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Logging Middleware (never log passwords, tokens, or auth headers)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({duration:.3f}s)"
    )
    return response


# Global Exception Handler for validation errors to ensure consistent {"detail": "..."} format
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Extract first error message cleanly
    errors = exc.errors()
    first_error = errors[0]["msg"] if errors else "Validation error"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": first_error},
    )


# Health endpoint
@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok"}


# Register Routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(conversations.router)
app.include_router(artifacts.router)
