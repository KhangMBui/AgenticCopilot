"""
ReAct agent endpoint.

This endpoint executes the manual ReAct agent (M4) and records the run.

It is separate from /chat because:
- /chat = RAG conversational interface
- /agent = tool-using agent trace (think: "developer / debug mode" or "copilot mode")
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from app.db import get_db
from app.models import Workspace, AgentRun, ToolCall
from app.schemas.agent import AgentRequest, AgentResponse, AgentStepResponse
from core.agents.ReAct.agent import ReActAgent
from core.llm import OpenAIClient
from core.embeddings import OpenAIEmbeddingsClient
from core.tools import ToolRegistry
from core.tools.builtins import RetrieveTool, CalculatorTool, FinalAnswerTool
from app.settings import settings

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("", response_model=AgentResponse)
def run_agent(request: AgentRequest, db: Session = Depends(get_db)):
    """
    Execute ReAct agent to answer a question and return the full trace.

    Agent's mental loop:
    - The LLM reads the question + tool list + history
    - Produces: Thought + Action + Action Input
    - We execute that tool and return Observation
    - Repeat until Action == final_answer or max steps hit

    What we store:
    - AgentRun: overall run metadata (query, status, final answer)
    - ToolCall: per-step record (thought, tool name, inputs, outputs)
    """

    # ---------------------------------------------------------------------
    # 0) Validate workspace exists (multi-tenant safety)
    # ---------------------------------------------------------------------
    workspace = db.get(Workspace, request.workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {request.workspace_id} not found",
        )

    # ---------------------------------------------------------------------
    # 1) Initialize core dependencies (LLM + embeddings)
    # ---------------------------------------------------------------------
    # llm = "brain": produces next steps
    llm = OpenAIClient(api_key=settings.openai_api_key)

    # embeddings = used by RetrieveTool (semantic search over pgvector)
    embeddings = OpenAIEmbeddingsClient(api_key=settings.openai_api_key)

    # ---------------------------------------------------------------------
    # 2) Create ToolRegistry and register tools the agent is allowed to use
    # ---------------------------------------------------------------------
    registry = ToolRegistry()

    # RetrieveTool is our semantic search tool:
    # it embeds a query and finds relevant chunks in DB for this workspace.
    # It needs access to db session + embeddings client + workspace boundary.
    registry.register(RetrieveTool(db, embeddings, request.workspace_id))

    # CalculatorTool is a deterministic arithmetic tool.
    registry.register(CalculatorTool())

    # FinalAnswerTool is usually a "pseudo-tool" that just returns the answer.
    # This makes it easy to detect termination: Action == "final_answer".
    registry.register(FinalAnswerTool())

    # ---------------------------------------------------------------------
    # 3) Create agent object and run it
    # ---------------------------------------------------------------------
    agent = ReActAgent(llm=llm, tools=registry, max_steps=10, verbose=True)

    # Execute agent loop and collect AgentState (history + status + final answer)
    state = agent.run(query=request.query)

    # ---------------------------------------------------------------------
    # 4) Store the run in DB (AgentRun row)
    # ---------------------------------------------------------------------
    agent_run = AgentRun(
        workspace_id=request.workspace_id,
        query=request.query,
        status=state.status,
        final_answer=state.final_answer,
        error=state.error,
        total_steps=len(state.steps),
        # completed_at should be set if run finished.
        # Some designs set completed_at for both completed and failed.
        completed_at=datetime.utcnow() if state.is_finished() else None,
    )
    db.add(agent_run)

    # flush() assigns agent_run.id so ToolCall rows can reference it
    db.flush()

    # ---------------------------------------------------------------------
    # 5) Store each step as a ToolCall row
    # ---------------------------------------------------------------------
    # Each AgentStep becomes a ToolCall record so you have a trace in Postgres.
    for step in state.steps:
        tool_call = ToolCall(
            agent_run_id=agent_run.id,
            step_number=step.step_number,
            # If step.action is None, record "none" so trace stays consistent.
            tool_name=step.action or "none",
            # Model's internal reasoning at that step
            thought=step.thought,
            # Parameters passed to tool
            input_params=step.action_input,
            # Observation = tool output or parse/tool error message
            output=step.observation,
            success=step.success,
        )
        db.add(tool_call)

    # Persist run + tool calls
    db.commit()
    db.refresh(agent_run)

    # ---------------------------------------------------------------------
    # 6) Build API response (AgentResponse)
    # ---------------------------------------------------------------------
    # Return:
    # - run metadata
    # - final answer
    # - step trace (AgentStepResponse list)
    return AgentResponse(
        run_id=agent_run.id,
        query=agent_run.query,
        status=agent_run.status,
        final_answer=agent_run.final_answer,
        error=agent_run.error,
        steps=[
            AgentStepResponse(
                step_number=s.step_number,
                thought=s.thought,
                action=s.action,
                action_input=s.action_input,
                observation=s.observation,
                is_final=s.is_final,
                success=s.success,
            )
            for s in state.steps
        ],
        total_steps=agent_run.total_steps,
        total_tokens=agent_run.total_tokens,
        created_at=agent_run.created_at,
        completed_at=agent_run.completed_at,
    )
