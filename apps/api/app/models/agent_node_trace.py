"""
Database model for storing per-node execution traces
from the multi-agent LangGraph workflow.

Each row represents one node execution event, such as:
- supervisor decision
- research worker run
- math worker run
- finish node execution

Why this table exists:
- lets us inspect how the agent made decisions
- helps with debugging failed runs
- supports observability / audit trail
- makes it easier to build dashboards later
"""

from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class AgentNodeTrace(Base):
    """
    Stores one trace event per graph node execution.

    Example:
    A single multi-agent run may produce rows like:

    step 0 -> supervisor -> route_to=research
    step 1 -> research   -> success=True
    step 2 -> supervisor -> route_to=math
    step 3 -> math       -> success=True
    step 4 -> finish     -> success=True
    """

    __tablename__ = "agent_node_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Foreign key to the parent AgentRun.
    # This connects each trace row to one full multi-agent run.
    # ondelete="CASCADE" means:
    # if the parent agent run is deleted, its trace rows are deleted too.
    agent_run_id: Mapped[int] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Which step in the run this node execution happened at.
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Name of the node that executed.
    # Example: "supervisor", "research", "math", "finish"
    node_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Type of trace event.
    # Right now default is "node_execution".
    # This leaves room for future trace types like:
    # "routing_decision", "tool_call", "retry", etc.
    event_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="node_execution"
    )

    # Whether this node execution succeeded.
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # If this trace came from the supervisor, this may record where it routed next.
    # Example: "research", "math", "finish"
    # For worker nodes, this may be None.
    route_to: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Flexible JSON column storing the raw trace payload.
    # Useful for debugging because it preserves full trace data.
    # Example payload:
    # {
    #   "node": "supervisor",
    #   "step": 2,
    #   "route_to": "math",
    #   "reason": "Need calculation"
    # }
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Timestamp when this trace row was created.
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
