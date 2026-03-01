"""
Pydantic schemas for document ingestion endpoints.
"""

from pydantic import BaseModel, Field
from datetime import datetime

# ===== REQUEST SCHEMAS =====


class DocumentCreateRequest(BaseModel):
    """Request to create a document."""

    filename: str = Field(..., max_length=500)
    content: str = Field(..., min_length=1)
    mime_type: str | None = Field(None, max_length=100)


class DocumentListQuery(BaseModel):
    """Query params for listing documents."""

    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


# ===== RESPONSE SCHEMAS =====


class ChunkResponse(BaseModel):
    """Single chunk response."""

    id: int
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    """Document response (without chunks)."""

    id: int
    workspace_id: int
    filename: str
    mime_type: str | None
    size_bytes: int | None
    chunk_count: int  # computed field
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentDetailResponse(DocumentResponse):
    """Document response with chunks included."""

    chunks: list[ChunkResponse] = []


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int
