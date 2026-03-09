from __future__ import annotations

from datetime import datetime
from sqlalchemy.orm import Session

from app.models import AgentRun, ToolCall
from app.models.agent_node_trace import AgentNodeTrace


def persist_multi_agent_result(
    db: Session,
    *,
    workspace_id: int,
    query: str,
    result: dict,
) -> AgentRun:
    """
    Persist a completed multi-agent run into the database.

    Stores 3 layers of information:

    1) AgentRun
       - overall run metadata
       - final answer / status / step count

    2) AgentNodeTrace
       - one row per graph node execution
       - supports step-by-step debugging and observability

    3) ToolCall
       - current approximation of worker lineage
       - stores which worker executed at which step and whether it succeeded

    Why this function exists:
    - separates persistence logic from graph execution logic
    - keeps endpoint/controller code cleaner
    - centralizes all DB writes for one agent run
    """

    # Create the main run-level DB object.
    # This represents the entire multi-agent execution as one record.
    agent_run = AgentRun(
        workspace_id=workspace_id,
        query=query,
        status=result.get("status", "completed"),
        final_answer=result.get("final_answer"),
        error=result.get("error"),
        total_steps=result.get("step_count", 0),
        total_tokens=None,
        completed_at=datetime.utcnow(),
    )
    db.add(agent_run)

    # Flush sends pending INSERT to DB without committing yet.
    # Important because we need agent_run.id right away
    # for related trace/tool rows.
    db.flush()

    # ---------------------------------------------------------
    # 1) Save node-level execution trace
    # ---------------------------------------------------------
    for t in result.get("trace", []):
        # Convert one trace dict from result["trace"]
        # into one AgentNodeTrace DB row.
        trace_row = AgentNodeTrace(
            agent_run_id=agent_run.id,
            step_number=int(t.get("step", 0)),
            node_name=str(t.get("node", "unknown")),
            event_type="node_execution",
            success=bool(t.get("success", True)),
            route_to=t.get("route_to"),
            payload=t,
        )
        db.add(trace_row)
    # ---------------------------------------------------------
    # 2) Save worker lineage as ToolCall rows
    # ---------------------------------------------------------
    # Current design note:
    # workers may internally call multiple tools,
    # but for now we only persist one lineage row per worker output.
    for w in result.get("worker_outputs", []):
        step = int(w.get("step", 0))
        worker_name = str(w.get("worker_name", "unknown_worker"))
        success = bool(w.get("success", False))

        # If worker succeeded, save its content.
        # If failed, save the error string or fallback message.
        output = w.get("content") if success else (w.get("error") or "worker failed")

        # Even though this is called ToolCall,
        # here it is being used more as a lineage/audit record
        # for worker executions.
        tool_call = ToolCall(
            agent_run_id=agent_run.id,
            step_number=step,
            tool_name=f"{worker_name}_worker",
            thought=f"Worker lineage event for {worker_name}",
            input_params={"lineage": {"worker": worker_name, "step": step}},
            output=output,
            success=success,
        )
        db.add(tool_call)

    # Commit all inserts:
    # - AgentRun
    # - AgentNodeTrace rows
    # - ToolCall rows
    db.commit()

    # Refresh from DB so returned object has committed values populated.
    db.refresh(agent_run)

    # Return the main AgentRun row for API response usage.
    return agent_run
