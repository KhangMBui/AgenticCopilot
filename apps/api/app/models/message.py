"""
Message model for chat messages.
"""

from sqlalchemy import String, Text, ForeignKey, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from app.base import Base


class Message(Base):
    """A single message in a conversation."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # 'user' or 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Metadata
    model: Mapped[str] = mapped_column(
        String(100), nullable=True
    )  # LLM model used (for assistant messages)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=True)

    # Citations (JSON array of chunk IDs that were used)
    cited_chunk_ids: Mapped[list] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
