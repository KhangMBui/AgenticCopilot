"""
Pydantic schemas for workspace endpoints
"""

from pydantic import BaseModel, Field
from datetime import datetime


class WorkspaceCreateRequest(BaseModel):
    """Request to create a workspace."""

    name: str = Field(..., min_length=1, max_length=200)


class WorkspaceResponse(BaseModel):
    """Workspace response."""

    id: int
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}
