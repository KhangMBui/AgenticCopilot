from datetime import datetime

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.base import Base


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    documents: Mapped[list["Document"]] = relationship(  # noqa: F821
        back_populates="workspace", cascade="all, delete-orphan"
    )
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
