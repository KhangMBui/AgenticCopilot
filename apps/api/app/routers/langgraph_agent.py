"""
LangGraph agent endpoint (for comparison with manual ReAct).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from app.db import get_db
from app.models import Workspace, AgentRun, ToolCall
from app.schemas.agent import AgentRequest, AgentResponse, AgentStepResponse
from core.agents.langgraph.agent import run_langgraph_agent
from core.embeddings import OpenAIEmbeddingsClient
from app.settings import settings


router = APIRouter(prefix="/agent/langgraph", tags=["agent-langgraph"])


@router.post("", response_model=AgentResponse)
def run_langgraph_agent_endpoint(request: AgentRequest, db: Session = Depends(get_db)):
    """
    Execute LangGraph ReAct agent (for comparison with manual implementation)

    This uses LangGraph's built-in:
    - Tool calling
    - State management
    - Execution graph

    Compare with /agent (manual) to see differences
    """

    # Validate workspace
    workspace = db.get(Workspace, request.workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {request.workspace_id} not found",
        )

    # Initialize embeddings
    embeddings = OpenAIEmbeddingsClient(api_key=settings.openai_api_key)

    # Run LangGraph agent
    result = run_langgraph_agent(
        query=request.query,
        api_key=settings.openai_api_key,
        db=db,
        embeddings=embeddings,
        workspace_id=request.workspace_id,
        max_steps=10,
    )

    # Store in DB (reusing same tables)
    agent_run = AgentRun(
        workspace_id=request.workspace_id,
        query=request.query,
        status=result["status"],
        final_answer=result["final_answer"],
        total_steps=result["total_steps"],
        completed_at=datetime.utcnow(),
    )
    db.add(agent_run)
    db.flush()

    # Store steps (simplified for LangGraph)
    for step_data in result["steps"]:
        if step_data["type"] == "tool_call":
            tool_call = ToolCall(
                agent_run_id=agent_run.id,
                step_number=step_data["step"],
                tool_name=step_data["tool"],
                thought="[LangGraph managed]",
                input_params=step_data["args"],
                output="[See messages]",
                success=True,
            )
            db.add(tool_call)
    db.commit()
    db.refresh(agent_run)

    # Build response
    return AgentResponse(
        run_id=agent_run.id,
        query=agent_run.query,
        status=agent_run.status,
        final_answer=agent_run.final_answer,
        error=agent_run.error,
        steps=[
            AgentStepResponse(
                step_number=s["step"],
                thought="[LangGraph auto-managed]",
                action=s.get("tool"),
                action_input=s.get("args"),
                observation=s.get("content", "[tool output]"),
                is_final=False,
                success=True,
            )
            for s in result["steps"]
        ],
        total_steps=agent_run.total_steps,
        total_tokens=None,  # LangGraph doesn't expose this easily
        created_at=agent_run.created_at,
        completed_at=agent_run.completed_at,
    )
