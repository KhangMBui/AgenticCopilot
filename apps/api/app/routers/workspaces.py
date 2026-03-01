"""
Workspace management endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.db import get_db
from app.models import Workspace
from app.schemas.workspaces import WorkspaceCreateRequest, WorkspaceResponse

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
def create_workspace(request: WorkspaceCreateRequest, db: Session = Depends(get_db)):
    """Create a new workspace."""
    workspace = Workspace(name=request.name)
    db.add(workspace)
    db.commit()
    db.refresh(workspace)
    return workspace


@router.get("", response_model=list[WorkspaceResponse])
def list_workspaces(db: Session = Depends(get_db)):
    """List all workspaces."""
    stmt = select(Workspace).order_by(Workspace.created_at.desc())
    workspaces = db.scalars(stmt).all()
    return workspaces


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace(workspace_id: int, db: Session = Depends(get_db)):
    """Get workspace by ID."""
    workspace = db.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )
    return workspace
