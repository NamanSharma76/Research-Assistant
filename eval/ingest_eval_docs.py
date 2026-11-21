"""Offline ingestion script to populate isolated evaluation data directory with the 3 PDFs."""
import os
import shutil
import sys
from pathlib import Path

eval_dir = Path(__file__).resolve().parent
repo_dir = eval_dir.parent
backend_dir = repo_dir / "academic-assistant" / "backend"

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

# Configure data directory to isolated location if not already specified
isolated_data_dir = eval_dir / "data"
os.environ.setdefault("DATA_DIR", str(isolated_data_dir))
os.environ.setdefault("CHROMA_DIR", str(isolated_data_dir / "chroma"))
os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", str(isolated_data_dir / "chroma"))
os.environ.setdefault("UPLOADS_DIR", str(isolated_data_dir / "uploads"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{isolated_data_dir / 'app.db'}")

from app.rag.ingest import ingest_pdf, get_chroma_client, compute_doc_id
from app.config import CHROMA_COLLECTION_NAME

SOURCE_PDFS = {
    "paper_a": backend_dir / "data" / "uploads" / "attention_is_all_you_need.pdf",
    "paper_b": backend_dir / "data" / "uploads" / "d7c2669d-42c5-45f4-886e-b981c1899fc0" / "062c183e-dc3f-42ef-a313-b37907f427ad.pdf",
    "paper_c": backend_dir / "data" / "uploads" / "d7c2669d-42c5-45f4-886e-b981c1899fc0" / "6d60bb5f-eb47-4a40-9c67-6d0e0bc0c557.pdf",
}


def setup_isolated_eval_data() -> dict:
    """Copies the 3 PDFs to eval/data/uploads and ingests them into eval/data/chroma."""
    eval_uploads = isolated_data_dir / "uploads"
    eval_chroma = isolated_data_dir / "chroma"
    eval_uploads.mkdir(parents=True, exist_ok=True)
    eval_chroma.mkdir(parents=True, exist_ok=True)

    ingested_info = {}

    print(f"=== SETTING UP ISOLATED EVAL DATA ===")
    print(f"Data Dir: {isolated_data_dir}")
    print(f"Chroma Dir: {eval_chroma}")
    print(f"Uploads Dir: {eval_uploads}\n")

    for key, src_path in SOURCE_PDFS.items():
        if not src_path.exists():
            raise FileNotFoundError(f"Source PDF not found at: {src_path}")

        dest_name = f"{key}_{src_path.name}"
        dest_path = eval_uploads / dest_name
        shutil.copy2(src_path, dest_path)

        doc_id = compute_doc_id(str(dest_path))
        print(f"Ingesting {key} ({dest_name}) -> doc_id: {doc_id}...")
        ingest_res = ingest_pdf(str(dest_path))
        num_chunks = ingest_res.get("chunks_count", 0)
        print(f"  -> Ingested {num_chunks} chunks.")

        ingested_info[key] = {
            "file": str(dest_path),
            "doc_id": doc_id,
            "chunks": num_chunks,
        }

    client = get_chroma_client()
    collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    print(f"\nVerification: Total chunks in isolated collection '{CHROMA_COLLECTION_NAME}': {collection.count()}")

    return ingested_info


if __name__ == "__main__":
    setup_isolated_eval_data()
