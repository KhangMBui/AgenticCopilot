"""
Pydantic schemas for search endpoints.
"""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search query."""

    query: str = Field(..., min_length=1, max_length=1000)
    workspace_id: int | None = None
    limit: int = Field(5, ge=1, le=50)


class SearchResult(BaseModel):
    """Single search result."""

    chunk_id: int
    document_id: int
    document_filename: str
    content: str
    score: float  # cosine similarity (0-1, higher is better)
    chunk_index: int


class SearchResponse(BaseModel):
    """Search results."""

    query: str
    results: list[SearchResult]
    total_results: int
