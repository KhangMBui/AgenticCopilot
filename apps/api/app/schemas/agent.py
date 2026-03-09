"""
Pydantic schemas for agent endpoints.
"""

from pydantic import BaseModel, Field
from datetime import datetime

# ===== REQUEST =====


class AgentRequest(BaseModel):
    """Agent query request."""

    query: str = Field(..., min_length=1, max_length=2000)
    workspace_id: int


# ===== RESPONSE =====


class AgentStepResponse(BaseModel):
    """Single agent step."""

    step_number: int
    thought: str
    action: str | None
    action_input: dict | None
    observation: str | None
    is_final: bool
    success: bool


class AgentResponse(BaseModel):
    """Agent execution response."""

    run_id: int
    query: str
    status: str
    # draft_asnwer: str | None
    final_answer: str | None
    error: str | None
    steps: list[AgentStepResponse]
    total_steps: int
    total_tokens: int | None
    created_at: datetime
    completed_at: datetime | None
