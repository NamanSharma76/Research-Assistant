"""Document management routes: upload, list, get, sections, delete."""
import os
import uuid
import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, Response
from sqlalchemy.orm import Session
from app.config import UPLOADS_DIR, MAX_UPLOAD_MB
from app.db.session import get_db
from app.db.models import User, Document
from app.api.schemas import DocumentResponse, DocumentSectionsResponse
from app.api.deps import get_current_user, get_owned_document
from app.rag.ingest import compute_doc_id, delete_document as delete_chroma_document
from app.rag.retrieval import list_doc_sections
from app.services.ingestion_service import run_ingestion_task
from app.agents.cache import CACHE_BASE_DIR

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    response: Response,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    # Stream file into user-specific folder while verifying size and %PDF magic bytes
    user_upload_dir = UPLOADS_DIR / user.id
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    temp_filename = f"{uuid.uuid4()}.pdf"
    file_path = user_upload_dir / temp_filename

    total_bytes = 0
    header_checked = False

    try:
        with open(file_path, "wb") as out_f:
            while chunk := await file.read(65536):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_MB}MB",
                    )
                if not header_checked and total_bytes >= 4:
                    # Check %PDF magic header
                    out_f.seek(0)
                    magic = chunk[:4] if total_bytes == len(chunk) else None
                    if magic is None or not magic.startswith(b"%PDF"):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Uploaded file is not a valid PDF document (missing %PDF header)",
                        )
                    header_checked = True
                out_f.write(chunk)
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise e

    # Compute doc_id from the saved PDF file
    doc_id = compute_doc_id(str(file_path))

    # Check if this user already uploaded this document
    existing = db.query(Document).filter(
        Document.user_id == user.id,
        Document.doc_id == doc_id,
    ).first()

    if existing:
        # Clean up the newly uploaded duplicate file
        if file_path.exists():
            file_path.unlink()
        # If existing document is ready but has num_chunks == 0, refresh from ChromaDB chunks count
        if existing.status == "ready" and existing.num_chunks == 0:
            try:
                from app.rag.retrieval import get_all_chunks
                chunks = get_all_chunks(doc_id)
                if chunks:
                    existing.num_chunks = len(chunks)
                    db.commit()
                    db.refresh(existing)
            except Exception:
                pass
        response.status_code = status.HTTP_200_OK
        return existing

    # Create new Document record with status "processing"
    doc_row = Document(
        user_id=user.id,
        doc_id=doc_id,
        filename=file.filename or "paper.pdf",
        file_path=str(file_path),
        status="processing",
        num_chunks=0,
    )
    db.add(doc_row)
    db.commit()
    db.refresh(doc_row)

    # Schedule background ingestion
    background_tasks.add_task(run_ingestion_task, doc_row.id, str(file_path))

    response.status_code = status.HTTP_202_ACCEPTED
    return doc_row


@router.get("", response_model=List[DocumentResponse])
def list_documents(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    docs = db.query(Document).filter(Document.user_id == user.id).all()
    return docs


@router.get("/{id}", response_model=DocumentResponse)
def get_document(
    doc: Document = Depends(get_owned_document),
):
    return doc


@router.get("/{id}/sections", response_model=DocumentSectionsResponse)
def get_document_sections(
    doc: Document = Depends(get_owned_document),
):
    if doc.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document is in status '{doc.status}'; sections are not available yet",
        )
    sections = list_doc_sections(doc.doc_id)
    return {"doc_id": doc.doc_id, "sections": sections}


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document_endpoint(
    doc: Document = Depends(get_owned_document),
    db: Session = Depends(get_db),
):
    doc_id_val = doc.doc_id
    file_path = Path(doc.file_path)

    # Delete row from DB (cascades to conversations and messages)
    db.delete(doc)
    db.commit()

    # Delete PDF file on disk
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass

    # Check if ANY other document row across any user still references doc_id_val
    remaining = db.query(Document).filter(Document.doc_id == doc_id_val).count()
    if remaining == 0:
        # Delete ChromaDB chunks
        try:
            delete_chroma_document(doc_id_val)
        except Exception:
            pass
        # Delete cache directory for doc_id
        cache_dir = CACHE_BASE_DIR / doc_id_val
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

    return None
