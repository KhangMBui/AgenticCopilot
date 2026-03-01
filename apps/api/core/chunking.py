"""
Simple character-based chunking with overlap.
Production systems use semantic chunking, but this is a solid M1 baselline
"""

from dataclasses import dataclass


@dataclass
class ChunkResult:
    """Single chunk result."""

    content: str
    chunk_index: int
    start_char: int
    end_char: int


def chunk_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[ChunkResult]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks

    Returns:
        List of ChunkResult objects
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        # Calculate end position
        end = start + chunk_size

        # Don't exceed text length
        if end > len(text):
            end = len(text)

        # Extract chunk content
        chunk_content = text[start:end].strip()

        # Only add non-empty chunks
        if chunk_content:
            chunks.append(
                ChunkResult(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_index += 1

        # # Move start position (with overlap), ensuring forward progress
        # next_start = end - chunk_overlap
        # if next_start <= start:
        #     break
        # start = next_start

        # If we've consumed all the text, we're done
        if end >= len(text):
            break

        # Move start position with overlap
        start = end - chunk_overlap

    return chunks
