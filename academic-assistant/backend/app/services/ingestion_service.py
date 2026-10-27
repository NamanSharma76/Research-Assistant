"""Background ingestion worker with dedicated DB sessions and global serialization lock."""
import threading
import logging
from app.db.session import SessionLocal
from app.db.models import Document
from app.rag.ingest import ingest_pdf

logger = logging.getLogger(__name__)

# Serialize all PDF ingestions: embedding models and ChromaDB writes are not thread-safe
_ingestion_lock = threading.Lock()


def run_ingestion_task(document_id: str, file_path: str) -> None:
    """Runs ingestion in background, acquires global lock, updates document status upon finish."""
    with _ingestion_lock:
        db = SessionLocal()
        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                logger.error(f"Document {document_id} not found for ingestion")
                return

            try:
                result = ingest_pdf(file_path)
                # ingest_pdf returns 'chunks_count' (or 'num_chunks' if already indexed/mocked)
                num_chunks = result.get("chunks_count", result.get("num_chunks", 0))
                doc.status = "ready"
                doc.num_chunks = num_chunks
                doc.error = None
                db.commit()
            except Exception as e:
                logger.exception(f"Ingestion failed for doc {document_id}: {e}")
                doc.status = "failed"
                doc.error = str(e)[:500]
                db.commit()
        finally:
            db.close()
