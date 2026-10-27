"""Artifact generation endpoints for summary, notes, and outline."""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.db.models import Document
from app.api.deps import get_owned_document
from app.agents.summarizer import summarize_document
from app.agents.notes import generate_notes
from app.agents.outline import generate_outline

router = APIRouter(prefix="/documents", tags=["artifacts"])


@router.get("/{id}/summary")
def get_document_summary(
    force: bool = Query(False, description="Force re-generation bypassing cache"),
    doc: Document = Depends(get_owned_document),
) -> Dict[str, Any]:
    if doc.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document status is '{doc.status}'; summaries require status 'ready'",
        )
    return summarize_document(doc.doc_id, force=force)


@router.get("/{id}/notes")
def get_document_notes(
    force: bool = Query(False, description="Force re-generation bypassing cache"),
    doc: Document = Depends(get_owned_document),
) -> Dict[str, Any]:
    if doc.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document status is '{doc.status}'; notes require status 'ready'",
        )
    return generate_notes(doc.doc_id, force=force)


@router.get("/{id}/outline")
def get_document_outline(
    slides: int = Query(10, description="Target slide count (clamped 5 to 15)"),
    force: bool = Query(False, description="Force re-generation bypassing cache"),
    doc: Document = Depends(get_owned_document),
) -> Dict[str, Any]:
    if doc.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document status is '{doc.status}'; outlines require status 'ready'",
        )
    clamped_slides = max(5, min(15, slides))
    return generate_outline(doc.doc_id, num_slides=clamped_slides, force=force)
