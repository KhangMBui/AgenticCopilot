"""
Milestone 6, phase 1: shared state + contracts for multi-agent LangGraph
"""

from __future__ import annotations
import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

RouteTarget = Literal["research", "math", "finish"]


# TypedDict means it will behave like a dictionary with {key: item, ...}
class WorkerOutput(TypedDict, total=False):
    """Standard output cnotract every worker must write"""

    worker_name: str
    success: bool
    content: str
    sources: list[str]
    error: str | None
    step: int


class SupervisorDecision(TypedDict):
    """Routing decision contract returned by supervisor logic"""

    route_to: RouteTarget
    reason: str


class MultiAgentState(TypedDict):
    """
    Shared graph state for M6 multi-agent orchestration
    """

    # Conversation history (LangGraph reducer appends); a list of LangChain's basemessage
    # A reducer means: When multiple nodes write to the same state field, LangGraph needs to know: “How do I combine the old value and new value?”
    # For messages, add_messages means: append/merge new messages into the existing message history
    # So if state currently has: messages = [HumanMessage("Hi")]
    # This syntax means: t his field has a normal Python type (list[BaseMsg]), plus extra metadata (add_messages)”
    # So if current state has messages = [HumanMessage("Hi")]
    # and a node returns {"messages": [AIMessage("Hello")]}
    # LangGraph merges them into:
    # messages = [
    # HumanMessage("Hi"),
    # AIMessage("Hello")
    # ]
    messages: Annotated[list[BaseMessage], add_messages]

    # User input
    query: str

    # Supervisor planning
    plan: str | None
    next_node: RouteTarget

    # Worker memory buckets
    research_notes: Annotated[list[str], operator.add]
    math_results: Annotated[list[str], operator.add]
    worker_outputs: Annotated[list[WorkerOutput], operator.add]

    # Observability/debug trace
    trace: Annotated[list[dict[str, Any]], operator.add]

    # Guardrails
    step_count: int
    max_steps: int

    # Finalization
    final_answer: str | None
    draft_answer: str | None
    error: str | None
