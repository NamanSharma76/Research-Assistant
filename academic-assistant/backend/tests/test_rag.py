"""Unit tests for Phase 1 RAG pipeline."""
import os
import shutil
import tempfile
import pytest
import fitz  # PyMuPDF
from pathlib import Path

from app.rag.ingest import (
    compute_doc_id,
    extract_pages_with_sections,
    chunk_extracted_blocks,
    ingest_pdf,
    delete_document,
    get_chroma_client,
)
from app.rag.retrieval import search, get_section, get_all_chunks
from app.rag.prompts import is_refusal
from app.config import CHROMA_COLLECTION_NAME


@pytest.fixture(scope="module")
def sample_pdf_path(tmp_path_factory):
    """Creates a temporary multi-page PDF with clear academic sections."""
    temp_dir = tmp_path_factory.mktemp("pdf_data")
    pdf_file = temp_dir / "test_paper.pdf"

    doc = fitz.open()

    # Page 1: Abstract & Introduction
    page1 = doc.new_page()
    page1.insert_text((50, 60), "Attention Is All You Need In Research", fontsize=16)
    page1.insert_text((50, 100), "Abstract", fontsize=14)
    page1.insert_text(
        (50, 130),
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. "
        "We introduce the Transformer, a novel network architecture based solely on attention mechanisms.",
        fontsize=10,
    )
    page1.insert_text((50, 220), "1. Introduction", fontsize=14)
    page1.insert_text(
        (50, 250),
        "Recurrent neural networks, especially LSTM and GRU, have been established as state of the art approaches. "
        "Recurrence introduces sequential computation constraints that prevent parallelization during training.",
        fontsize=10,
    )

    # Page 2: Model Architecture & References
    page2 = doc.new_page()
    page2.insert_text((50, 60), "2. Model Architecture", fontsize=14)
    page2.insert_text(
        (50, 90),
        "Most competitive neural sequence models have an encoder-decoder structure. "
        "Here, the encoder maps an input sequence of symbol representations to a sequence of continuous representations.",
        fontsize=10,
    )
    page2.insert_text((50, 200), "References", fontsize=14)
    page2.insert_text(
        (50, 230),
        "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones. Attention is all you need. In NeurIPS 2017.",
        fontsize=10,
    )

    doc.save(str(pdf_file))
    doc.close()

    yield str(pdf_file)


def test_chunking_keeps_metadata(sample_pdf_path):
    """Test that chunking preserves all required metadata fields."""
    doc_id = compute_doc_id(sample_pdf_path)
    blocks = extract_pages_with_sections(sample_pdf_path)
    chunks = chunk_extracted_blocks(blocks, doc_id=doc_id, filename="test_paper.pdf")

    assert len(chunks) > 0

    required_keys = {
        "doc_id",
        "filename",
        "section",
        "page_start",
        "page_end",
        "chunk_index",
        "chunk_id",
        "text",
    }

    for idx, chunk in enumerate(chunks):
        assert required_keys.issubset(chunk.keys())
        assert chunk["doc_id"] == doc_id
        assert chunk["filename"] == "test_paper.pdf"
        assert chunk["chunk_index"] == idx
        assert chunk["chunk_id"] == f"{doc_id}_c{idx:04d}"
        assert chunk["page_start"] in [1, 2]
        assert chunk["page_end"] in [1, 2]
        assert chunk["page_start"] <= chunk["page_end"]


def test_reingest_does_not_duplicate(sample_pdf_path):
    """Test that re-ingesting the exact same file does not create duplicate chunks."""
    result1 = ingest_pdf(sample_pdf_path)
    doc_id = result1["doc_id"]

    all_chunks_first = get_all_chunks(doc_id)
    count_first = len(all_chunks_first)
    assert count_first > 0

    # Second ingestion attempt of the same document
    result2 = ingest_pdf(sample_pdf_path)
    assert result2["status"] == "already_indexed"
    assert result2["doc_id"] == doc_id

    all_chunks_second = get_all_chunks(doc_id)
    assert len(all_chunks_second) == count_first


def test_get_section_returns_ordered_chunks(sample_pdf_path):
    """Test that get_section returns chunks filtered by section and strictly ordered by chunk_index."""
    doc_id = compute_doc_id(sample_pdf_path)
    section_chunks = get_section(doc_id, "Abstract")

    assert len(section_chunks) > 0
    indices = [c["chunk_index"] for c in section_chunks]
    assert indices == sorted(indices)
    for c in section_chunks:
        assert "abstract" in c["section"].lower()
        assert c["doc_id"] == doc_id


def test_search_respects_doc_id(sample_pdf_path, tmp_path_factory):
    """Test that semantic search strictly respects doc_id isolation and returns normalized scores."""
    # Create and ingest a second PDF with different content
    temp_dir = tmp_path_factory.mktemp("second_pdf")
    second_pdf = temp_dir / "biology_paper.pdf"

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 60), "Cellular Biology and Photosynthesis", fontsize=14)
    page.insert_text(
        (50, 100),
        "Chloroplasts capture light energy from the sun to produce glucose through photosynthesis in plants.",
        fontsize=10,
    )
    doc.save(str(second_pdf))
    doc.close()

    result2 = ingest_pdf(str(second_pdf))
    doc_id2 = result2["doc_id"]
    doc_id1 = compute_doc_id(sample_pdf_path)

    # Search in doc_id1 for transformers
    results_doc1 = search(doc_id=doc_id1, query="Transformer network architecture attention", k=3)
    assert len(results_doc1) > 0
    for res in results_doc1:
        assert 0.0 <= res["score"] <= 1.0
        assert "photosynthesis" not in res["text"].lower()

    # Search in doc_id2 for photosynthesis
    results_doc2 = search(doc_id=doc_id2, query="photosynthesis chloroplasts", k=3)
    assert len(results_doc2) > 0
    for res in results_doc2:
        assert 0.0 <= res["score"] <= 1.0
        assert "transformer" not in res["text"].lower()


def test_search_excludes_references_by_default(sample_pdf_path):
    """Test that search() excludes References by default and includes them when exclude_sections=()."""
    doc_id = compute_doc_id(sample_pdf_path)

    # Default search excludes References
    default_results = search(doc_id=doc_id, query="Ashish Vaswani NeurIPS", k=5)
    sections = [r["section"] for r in default_results]
    assert "References" not in sections

    # When exclude_sections=(), References are included
    all_results = search(doc_id=doc_id, query="Ashish Vaswani NeurIPS", k=5, exclude_sections=())
    all_sections = [r["section"] for r in all_results]
    assert "References" in all_sections


def test_is_refusal():
    """Test is_refusal() on exact sentences, casing, punctuation, and normal answers."""
    assert is_refusal("The paper does not appear to cover this.")
    assert is_refusal("the paper does not appear to cover this")
    assert is_refusal("THE PAPER DOES NOT APPEAR TO COVER THIS!")
    assert is_refusal("The paper does not appear to cover this because...")
    assert not is_refusal("The model achieves 28.4 BLEU on English-to-German [6 Results, pp. 8-9].")
    assert not is_refusal("Transformer is an attention-based model.")


def test_delete_document(tmp_path_factory):
    """Test that delete_document removes all chunks of a doc."""
    temp_dir = tmp_path_factory.mktemp("delete_test")
    pdf_file = temp_dir / "to_delete.pdf"

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 60), "Temporary Document For Deletion", fontsize=14)
    doc.save(str(pdf_file))
    doc.close()

    res = ingest_pdf(str(pdf_file))
    doc_id = res["doc_id"]
    assert len(get_all_chunks(doc_id)) > 0

    deleted_count = delete_document(doc_id)
    assert deleted_count > 0
    assert len(get_all_chunks(doc_id)) == 0
