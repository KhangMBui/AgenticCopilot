from __future__ import annotations
from typing import List, Tuple
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document
from core.embeddings import OpenAIEmbeddingsClient


def _split_text_with_offsets(
    text: str, chunk_size: int = 800, overlap: int = 120
) -> List[Tuple[str, int, int]]:
    text = (text or "").strip()
    if not text:
        return []

    out: List[Tuple[str, int, int]] = []
    step = max(1, chunk_size - overlap)
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        out.append((text[start:end], start, end))
        if end >= n:
            break
        start += step

    return out


def ingest_document_chunks(
    db: Session,
    *,
    document: Document,
    embeddings: OpenAIEmbeddingsClient,
    chunk_size: int = 800,
    overlap: int = 120,
) -> int:
    parts = _split_text_with_offsets(
        document.content, chunk_size=chunk_size, overlap=overlap
    )

    inserted = 0
    for idx, (part, start_char, end_char) in enumerate(parts):
        vector = embeddings.embed_text(part)
        row = Chunk(
            document_id=document.id,
            content=part,
            chunk_index=idx,
            start_char=start_char,  # <- required
            end_char=end_char,  # <- required
            embedding=vector,
        )
        db.add(row)
        inserted += 1

    return inserted
