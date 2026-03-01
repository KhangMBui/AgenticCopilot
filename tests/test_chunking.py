"""
Tests for chunking logic.
"""

from core.chunking import chunk_text


def test_chunk_text_basic():
    """Test basic chunking."""
    text = "a" * 2500  # 2500 chars
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 3  # should create 3 chunks
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 1000


def test_chunk_text_empty():
    """Test empty input."""
    chunks = chunk_text("")
    assert len(chunks) == 0


def test_chunk_text_small():
    """Test small text (no chunking needed)."""
    text = "Small text."
    chunks = chunk_text(text, chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].content == "Small text."
