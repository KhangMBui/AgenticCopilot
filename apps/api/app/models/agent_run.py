"""
Agent run tracking model.

Why this exists:
- Agents are not single-shot responses; they take multiple steps.
- In production, you must be able to inspect what happened:
    - What tools were called?
    - In what order?
    - With what inputs?
    - What did tools return?
    - Did it fail because of parsing, tool error, or max steps?

So we store:
- One row per agent run (AgentRun)
- Many rows per run for each tool call / step (ToolCall)
"""

from sqlalchemy import String, Text, ForeignKey, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from app.base import Base


class AgentRun(Base):
    """
    Track a single agent execution.

    One AgentRun corresponds to one user query handled by an agent.

    Example:
    - query: "Summarize this PDF and compute total cost"
    - status: "completed"
    - final_answer: "...summary..."
    - tool_calls: list of ToolCall rows (retrieve -> calculator -> final_answer)
    """

    __tablename__ = "agent_runs"

    # Primary key for the run
    id: Mapped[int] = mapped_column(primary_key=True)

    # Multi-tenant boundary:
    # every run belongs to one workspace.
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE")
    )

    # The original user query/prompt sent to the agent
    query: Mapped[str] = mapped_column(Text, nullable=False)

    # High-level run status.
    # Suggested values: "completed", "failed", "max_steps"
    status: Mapped[str] = mapped_column(String(20), nullable=False)

    # The final answer produced (if completed)
    final_answer: Mapped[str] = mapped_column(Text, nullable=True)

    # Error message (if failed or max_steps)
    error: Mapped[str] = mapped_column(Text, nullable=True)

    # How many steps were taken (length of state.steps)
    total_steps: Mapped[int] = mapped_column(Integer, nullable=False)

    # Optional usage tracking (LLM cost observability).
    # You'll likely fill this later from your LLMClient response.
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    completed_at: Mapped[datetime] = mapped_column(nullable=True)

    # Relationships
    # - workspace: ORM link to Workspace row
    # - tool_calls: list of ToolCall rows for this run
    workspace: Mapped["Workspace"] = relationship()
    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="agent_run",
        cascade="all, delete-orphan",  # deleting a run deletes its tool calls
    )


class ToolCall(Base):
    """
    Track one step / tool call inside an agent run.

    Even if the agent step did not call a tool, you can store "none" for tool_name
    so the UI trace is consistent.

    This is basically your "agent execution trace" table.
    """

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Many tool calls belong to one AgentRun.
    agent_run_id: Mapped[int] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE")
    )

    # Which step number in the agent loop this corresponds to.
    # Helps reconstruct order.
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Name of the tool the agent chose (e.g., "retrieve", "calculator", "final_answer")
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # The model's reasoning ("Thought") at that step.
    # Useful for debugging, but be careful with sensitive data in production logs.
    thought: Mapped[str] = mapped_column(Text, nullable=False)

    # Tool input parameters in JSON form.
    # Example: {"query": "...", "limit": 5}
    input_params: Mapped[dict] = mapped_column(JSON, nullable=True)

    # Tool output text (observation).
    # Example: "Found 5 chunks..." or "42"
    output: Mapped[str] = mapped_column(Text, nullable=True)

    # Whether tool call succeeded.
    # In production, you should set this based on ToolResult.success, not string matching.
    success: Mapped[bool] = mapped_column(nullable=False)

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationship back to AgentRun
    agent_run: Mapped["AgentRun"] = relationship(back_populates="tool_calls")
