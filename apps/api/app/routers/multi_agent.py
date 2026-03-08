"""
M6 Phase 6: Multi-agent orchestration endpoint

Purpose:
---------
Expose the LangGraph multi-agent system through a FastAPI route.

This endpoint:
1. validates the workspace
2. runs the multi-agent graph
3. persists the full run to the database
4. returns a structured API response
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Workspace
from app.schemas.agent import AgentRequest, AgentResponse, AgentStepResponse
from core.agents.langgraph_multi.graph import run_multi_agent
from core.agents.langgraph_multi.persistence import persist_multi_agent_result
from core.embeddings import OpenAIEmbeddingsClient
from app.settings import settings


router = APIRouter(prefix="/agent/multi", tags=["agent-multi"])


@router.post("", response_model=AgentResponse)
def run_multi_agent_endpoint(
    request: AgentRequest,
    db: Session = Depends(get_db),
):
    """
    Execute the multi-agent workflow for one request.

    Current behavior:
    - validates workspace existence
    - initializes embeddings client
    - runs the LangGraph multi-agent flow
    - persists run history to DB
    - returns final answer + step trace

    High-level routing behavior:
    - math-like questions may go to math worker
    - non-math questions may go to research worker
    - supervisor decides when to stop
    """

    # ---------------------------------------------------------
    # 1) Validate workspace
    # ---------------------------------------------------------
    workspace = db.query(Workspace).filter(Workspace.id == request.workspace_id).first()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {request.workspace_id} not found",
        )

    # ---------------------------------------------------------
    # 2) Initialize embeddings client
    # ---------------------------------------------------------
    # Research worker likely needs this for retrieval / semantic search.
    embeddings = OpenAIEmbeddingsClient(settings.openai_api_key)

    # ---------------------------------------------------------
    # 3) Run multi-agent graph
    # ---------------------------------------------------------
    # Executes the full supervisor + worker loop.
    result = run_multi_agent(
        query=request.query,
        db=db,
        embeddings=embeddings,
        workspace_id=request.workspace_id,
        api_key=settings.openai_api_key,
        max_steps=8,
    )

    # ---------------------------------------------------------
    # 4) Persist completed run to DB
    # ---------------------------------------------------------
    agent_run = persist_multi_agent_result(
        db=db, workspace_id=request.workspace_id, query=request.query, result=result
    )

    # ---------------------------------------------------------
    # 5) Build API step response objects from trace
    # ---------------------------------------------------------
    steps = []
    for i, trace_entry in enumerate(result.get("trace", [])):
        # Convert internal trace dict into a clean API response object.
        step = AgentStepResponse(
            step_number=trace_entry.get("step", i),
            thought=f"Node: {trace_entry.get('node', 'unknown')}",
            action=trace_entry.get("node"),
            action_input=trace_entry,
            observation=trace_entry.get("reason", "[node executed]"),
            is_final=trace_entry.get("node") == "finish",
            success=trace_entry.get("success", True),
        )
        steps.append(step)

    # ---------------------------------------------------------
    # 6) Return final API response
    # ---------------------------------------------------------
    return AgentResponse(
        run_id=agent_run.id,
        query=agent_run.query,
        status=agent_run.status,
        final_answer=agent_run.final_answer,
        error=agent_run.error,
        steps=steps,
        total_steps=agent_run.total_steps,
        total_tokens=agent_run.total_tokens,
        created_at=agent_run.created_at,
        completed_at=agent_run.completed_at,
    )
