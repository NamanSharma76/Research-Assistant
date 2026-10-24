"""PDF ingestion pipeline: PDF -> chunks -> embeddings -> ChromaDB."""
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import fitz  # PyMuPDF
import tiktoken
import chromadb
from chromadb.config import Settings

from app.config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
)
from app.rag.embeddings import get_embedding_model

# Tokenizer for estimating token counts
_ENCODER = tiktoken.get_encoding("cl100k_base")

# Known academic section names regex
KNOWN_SECTIONS = [
    r"abstract",
    r"introduction",
    r"background",
    r"related\s+work",
    r"overview",
    r"preliminaries",
    r"system\s+design",
    r"architecture",
    r"methodology",
    r"method",
    r"methods",
    r"approach",
    r"implementation",
    r"evaluation",
    r"experiments?",
    r"experimental\s+setup",
    r"results?",
    r"discussion",
    r"limitations?",
    r"future\s+work",
    r"conclusion",
    r"conclusions",
    r"acknowledg(?:e)?ments?",
    r"references",
    r"appendix",
]

NUMBERED_HEADING_REGEX = re.compile(
    r"^(?:(?:\d+\.)+\s*|[I|V|X]+\.?\s*|\d+\.?\s+)([A-Z][A-Za-z0-9\s\-_:,]{1,60})$"
)

SECTION_REGEXES = [re.compile(rf"^\s*(?:\d+\.?\s*)?{s}\s*$", re.IGNORECASE) for s in KNOWN_SECTIONS]


def compute_doc_id(file_path: str) -> str:
    """Computes a short 12-char SHA-256 hash of the file contents.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        12-character hex digest string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def count_tokens(text: str) -> int:
    """Counts tokens in a given string using cl100k_base encoding."""
    return len(_ENCODER.encode(text))


def _detect_heading(line: str, font_size: float, is_bold: bool, median_font_size: float) -> str | None:
    """Detects if a given line is a section heading based on formatting or regex.

    Returns normalized title if detected, else None.
    """
    clean = line.strip()
    if not clean or len(clean) > 80:
        return None

    # Reject bare numbers or punctuation
    if clean.isdigit() or re.match(r"^(?:\d+\.?)+$", clean):
        return None

    # Check against known academic sections
    for pattern in SECTION_REGEXES:
        if pattern.match(clean):
            return clean.title()

    # Check numbered headings: e.g. "1. Introduction", "1 Introduction", "3.2 Model Architecture"
    m = NUMBERED_HEADING_REGEX.match(clean)
    if m:
        if not clean.endswith((".", ":", ";")) and len(clean.split()) <= 10:
            return clean.title()

    # Check font characteristics if font is significantly larger or bold
    if (font_size > median_font_size * 1.15 or (is_bold and font_size >= median_font_size * 0.98)) and len(clean.split()) <= 6:
        if not clean.endswith((".", ":", ";", ",")):
            for s in KNOWN_SECTIONS:
                if re.search(rf"^\s*(?:\d+\.?\s*)?{s}\s*$", clean, re.IGNORECASE):
                    return clean.title()

    return None


def extract_pages_with_sections(pdf_path: str) -> List[Dict[str, Any]]:
    """Extracts text elements and page-level section tracking from PDF.

    Returns:
        List of blocks: [{'text': str, 'page': int, 'section': str}]
    """
    doc = fitz.open(pdf_path)
    all_font_sizes = []

    # First pass: collect font sizes to determine body text median
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            if b.get("type") == 0:  # text block
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        all_font_sizes.append(span.get("size", 10.0))

    median_font_size = 10.0
    if all_font_sizes:
        all_font_sizes.sort()
        median_font_size = all_font_sizes[len(all_font_sizes) // 2]

    current_section = "Unknown"
    extracted_blocks: List[Dict[str, Any]] = []

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])

        for b in blocks:
            if b.get("type") != 0:
                continue

            lines = b.get("lines", [])
            lines_text = ["".join(s.get("text", "") for s in l.get("spans", [])).strip() for l in lines]
            block_text = " ".join(lines_text).strip()
            if not block_text:
                continue

            all_spans = [s for l in lines for s in l.get("spans", [])]
            max_size = max((s.get("size", 10.0) for s in all_spans), default=10.0)
            is_bold = any(
                "bold" in s.get("font", "").lower()
                or "medi" in s.get("font", "").lower()
                or (s.get("flags", 0) & 2 != 0)
                for s in all_spans
            )

            # Test heading on entire block text first (e.g. "1 Introduction", "6 Results")
            detected_heading = _detect_heading(block_text, max_size, is_bold, median_font_size)

            # If not detected on entire block, test first line (e.g. "Acknowledgements")
            if not detected_heading and lines_text:
                first_line = lines_text[0]
                first_spans = lines[0].get("spans", [])
                first_size = max((s.get("size", 10.0) for s in first_spans), default=10.0)
                first_bold = any(
                    "bold" in s.get("font", "").lower()
                    or "medi" in s.get("font", "").lower()
                    or (s.get("flags", 0) & 2 != 0)
                    for s in first_spans
                )
                detected_heading = _detect_heading(first_line, first_size, first_bold, median_font_size)

            if detected_heading:
                current_section = detected_heading

            extracted_blocks.append({
                "text": block_text,
                "page": page_num,
                "section": current_section,
                "is_heading": bool(detected_heading),
            })

    doc.close()
    return extracted_blocks


def chunk_extracted_blocks(
    blocks: List[Dict[str, Any]],
    doc_id: str,
    filename: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """Chunks text while respecting section boundaries and token constraints.

    Args:
        blocks: Parsed blocks from PDF.
        doc_id: Unique document hash.
        filename: Original file name.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Token overlap between consecutive chunks.

    Returns:
        List of chunk dicts matching the required metadata schema.
    """
    if not blocks:
        return []

    # Group consecutive blocks by section
    section_groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    current_sec = blocks[0]["section"]
    current_list: List[Dict[str, Any]] = []

    for b in blocks:
        if b["section"] != current_sec and current_list:
            section_groups.append((current_sec, current_list))
            current_sec = b["section"]
            current_list = [b]
        else:
            current_list.append(b)
    if current_list:
        section_groups.append((current_sec, current_list))

    chunks: List[Dict[str, Any]] = []
    chunk_index = 0

    for section_name, sec_blocks in section_groups:
        current_tokens: List[int] = []
        current_pages: List[int] = []
        token_page_map: List[int] = []  # tracks page number for each token

        for b in sec_blocks:
            b_tokens = _ENCODER.encode(b["text"] + "\n\n")
            b_page = b["page"]
            current_tokens.extend(b_tokens)
            token_page_map.extend([b_page] * len(b_tokens))

        # Split current section tokens into chunks with overlap
        start_idx = 0
        total_tokens = len(current_tokens)

        if total_tokens == 0:
            continue

        while start_idx < total_tokens:
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk_slice = current_tokens[start_idx:end_idx]
            pages_slice = token_page_map[start_idx:end_idx]

            chunk_text = _ENCODER.decode(chunk_slice).strip()
            page_start = pages_slice[0] if pages_slice else 1
            page_end = pages_slice[-1] if pages_slice else page_start

            chunk_id = f"{doc_id}_c{chunk_index:04d}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "filename": filename,
                "section": section_name,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_index": chunk_index,
                "text": chunk_text,
            })
            chunk_index += 1

            if end_idx >= total_tokens:
                break
            # Step forward by chunk_size - chunk_overlap
            step = max(1, chunk_size - chunk_overlap)
            start_idx += step

    return chunks


def get_chroma_client() -> chromadb.ClientAPI:
    """Returns persistent ChromaDB client."""
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)


def delete_document(doc_id: str) -> int:
    """Deletes all chunks belonging to doc_id from ChromaDB collection.

    Args:
        doc_id: 12-character document identifier.

    Returns:
        Number of chunks deleted.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    existing = collection.get(where={"doc_id": doc_id})
    chunk_count = len(existing["ids"]) if existing and existing["ids"] else 0
    if chunk_count > 0:
        collection.delete(where={"doc_id": doc_id})
    return chunk_count


def ingest_pdf(pdf_path: str, force: bool = False) -> Dict[str, Any]:
    """Ingests a PDF file into ChromaDB idempotently.

    Args:
        pdf_path: Path to the PDF file.
        force: If True, deletes existing chunks for this doc_id before re-ingesting.

    Returns:
        Dict with status, doc_id, filename, and number of chunks.
    """
    p = Path(pdf_path)
    if not p.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    filename = p.name
    doc_id = compute_doc_id(str(p))

    client = get_chroma_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    # Check if doc_id is already ingested (idempotency check)
    existing = collection.get(where={"doc_id": doc_id}, limit=1)
    if existing and existing["ids"]:
        if force:
            delete_document(doc_id)
        else:
            existing_count = len(collection.get(where={"doc_id": doc_id})["ids"])
            return {
                "status": "already_indexed",
                "doc_id": doc_id,
                "filename": filename,
                "chunks_count": existing_count,
            }

    # Extract & Chunk
    blocks = extract_pages_with_sections(str(p))
    chunks = chunk_extracted_blocks(blocks, doc_id, filename)

    if not chunks:
        return {
            "status": "empty",
            "doc_id": doc_id,
            "filename": filename,
            "chunks_count": 0,
        }

    # Embed chunks
    embedder = get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_documents(texts)

    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "doc_id": c["doc_id"],
            "filename": c["filename"],
            "section": c["section"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "chunk_index": c["chunk_index"],
            "chunk_id": c["chunk_id"],
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    return {
        "status": "ingested",
        "doc_id": doc_id,
        "filename": filename,
        "chunks_count": len(chunks),
    }
