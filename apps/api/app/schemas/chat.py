"""
Pydantic schemas for chat endpoints.
"""

from pydantic import BaseModel, Field
from datetime import datetime


# ===== REQUEST SCHEMAS =====
class ChatRequest(BaseModel):
    """Chat message request."""

    message: str = Field(..., min_length=1, max_length=5000)
    conversation_id: int | None = None  # None = start new conversation
    workspace_id: int


# ===== RESPONSE SCHEMAS =====
class CitedSource(BaseModel):
    """A cited source chunk."""

    chunk_id: int
    document_filename: str
    content: str
    chunk_index: int
    relevance_score: float


class MessageResponse(BaseModel):
    """Single message."""

    id: int
    role: str
    content: str
    created_at: datetime

    # Assistant message metadata
    model: str | None = None
    total_tokens: int | None = None
    cited_sources: list[CitedSource] = []

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    """Conversation details."""

    id: int
    workspace_id: int
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int

    model_config = {"from_attributes": True}


class ChatResponse(BaseModel):
    """Chat response with answer and sources."""

    conversation_id: int
    message: MessageResponse
    sources: list[CitedSource]
