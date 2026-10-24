"""Retrieval functions for academic documents stored in ChromaDB."""
from typing import List, Dict, Any
from app.rag.ingest import get_chroma_client
from app.rag.embeddings import get_embedding_model
from app.config import CHROMA_COLLECTION_NAME


def search(
    doc_id: str,
    query: str,
    k: int = 5,
    exclude_sections: tuple = ("References",),
) -> List[Dict[str, Any]]:
    """Performs semantic search restricted to a specific doc_id.

    Args:
        doc_id: Document hash identifier.
        query: Query string.
        k: Maximum number of chunks to return.
        exclude_sections: Tuple of section names to exclude from search results.
                          Defaults to ("References",).

    Returns:
        List of dicts: {text, section, page_start, page_end, score}
        sorted descending by score.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    embedder = get_embedding_model()
    query_vector = embedder.embed_query(query)

    # Build Chroma metadata filter
    if exclude_sections:
        if len(exclude_sections) == 1:
            where_filter = {
                "$and": [
                    {"doc_id": doc_id},
                    {"section": {"$ne": exclude_sections[0]}},
                ]
            }
        else:
            where_filter = {
                "$and": [
                    {"doc_id": doc_id},
                    {"$and": [{"section": {"$ne": sec}} for sec in exclude_sections]},
                ]
            }
    else:
        where_filter = {"doc_id": doc_id}

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks: List[Dict[str, Any]] = []
    if not results or not results["documents"] or not results["documents"][0]:
        return chunks

    docs = results["documents"][0]
    metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
    distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

    for doc_text, meta, dist in zip(docs, metas, distances):
        # ChromaDB default cosine distance: range [0, 2], where 0 is identical
        # Normalized similarity score: 1 - (dist / 2.0)
        norm_score = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
        chunks.append({
            "text": doc_text,
            "section": meta.get("section", "Unknown"),
            "page_start": meta.get("page_start", 1),
            "page_end": meta.get("page_end", 1),
            "score": round(norm_score, 4),
        })

    # Sort descending by score
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def get_section(doc_id: str, section_name: str) -> List[Dict[str, Any]]:
    """Retrieves all chunks belonging to a section ordered by chunk_index.

    Args:
        doc_id: Document hash identifier.
        section_name: Name of the section (case-insensitive substring or exact match).

    Returns:
        List of chunk dicts ordered by chunk_index.
    """
    all_chunks = get_all_chunks(doc_id)
    # Filter by section (case-insensitive comparison or exact match)
    sec_lower = section_name.strip().lower()
    matching_chunks = [
        c for c in all_chunks
        if c.get("section", "").strip().lower() == sec_lower
        or sec_lower in c.get("section", "").strip().lower()
    ]
    return matching_chunks


def get_all_chunks(doc_id: str) -> List[Dict[str, Any]]:
    """Retrieves every chunk of the document ordered by chunk_index.

    Args:
        doc_id: Document hash identifier.

    Returns:
        Complete list of chunk dicts ordered by chunk_index.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    results = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"],
    )

    if not results or not results["documents"]:
        return []

    chunks: List[Dict[str, Any]] = []
    for doc_text, meta in zip(results["documents"], results["metadatas"]):
        chunks.append({
            "chunk_id": meta.get("chunk_id", ""),
            "doc_id": meta.get("doc_id", doc_id),
            "filename": meta.get("filename", ""),
            "section": meta.get("section", "Unknown"),
            "page_start": meta.get("page_start", 1),
            "page_end": meta.get("page_end", 1),
            "chunk_index": meta.get("chunk_index", 0),
            "text": doc_text,
        })

    # Sort strictly by chunk_index
    chunks.sort(key=lambda x: x["chunk_index"])
    return chunks


def list_doc_sections(doc_id: str) -> List[str]:
    """Returns distinct section names stored in ChromaDB for a doc_id in chunk appearance order."""
    all_chunks = get_all_chunks(doc_id)
    sections: List[str] = []
    seen = set()
    for c in all_chunks:
        sec = c.get("section", "Unknown").strip()
        if sec and sec not in seen:
            seen.add(sec)
            sections.append(sec)
    return sections
