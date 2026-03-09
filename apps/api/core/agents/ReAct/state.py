"""
ReAct agent state.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AgentStep:
    """Single step in agent execution."""

    step_number: int
    thought: str
    action: str | None = None
    action_input: dict | None = None
    observation: str | None = None
    is_final: bool = False
    success: bool = True  # ← NEW: track if tool execution succeeded


@dataclass
class AgentState:
    """Current state of agent execution"""

    query: str
    status: Literal["running", "completed", "failed", "max_steps"] = "running"
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    error: str | None = None

    def add_step(self, step: AgentStep):
        """Add a new step to history."""
        self.steps.append(step)

    def current_step_number(self) -> int:
        """Get current step number."""
        return len(self.steps) + 1

    def is_finished(self) -> bool:
        """Check if execution is complete."""
        return self.status != "running"
