"""
M6 Phase 1: Deterministic supervisor routing rules
(Phase 3 can replace/augment this with LLM-based supervisor)
"""

from __future__ import annotations
import re
from typing import Literal
from core.agents.langgraph_multi.state import MultiAgentState, SupervisorDecision

_MATH_HINTS = (
    "calculate",
    "sum",
    "total",
    "average",
    "percent",
    "ratio",
    "how many",
    "per day",
    "per week",
)


def _looks_math_query(query: str) -> bool:
    q = query.lower()
    if any(h in q for h in _MATH_HINTS):
        return True
    # any digit or arithmetic symbol
    return bool(re.search(r"[\d+\-*/()%]", q))


def decide_route(state: MultiAgentState) -> SupervisorDecision:
    """
    Contract-only routing logic for Phase 1.
    """
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 8)

    if step_count >= max_steps:
        return {"route_to": "finish", "reason": "Reached max_steps guardrail"}

    failed_research_attempts = sum(
        1
        for w in state.get("worker_outputs", [])
        if w.get("worker_name") == "research" and not w.get("success", False)
    )

    if failed_research_attempts >= 2:
        return {
            "route_to": "finish",
            "reason": "Research failed repeatedly; finishing with grounded fallback",
        }

    query = state["query"]
    # has_research = len(state.get("research_notes", [])) > 0
    has_math = len(state.get("math_results", [])) > 0
    needs_math = _looks_math_query(query)

    has_grounded_research = any(
        w.get("worker_name") == "research"
        and w.get("success") is True
        and (w.get("content") or "").strip()
        for w in state.get("worker_outputs", [])
    )

    if needs_math and not has_math:
        return {
            "route_to": "math",
            "reason": "Query appears math-heavy and no math result yet",
        }

    if not has_grounded_research:
        return {
            "route_to": "research",
            "reason": "Need grounded retrieval before final answer",
        }

    return {"route_to": "finish", "reason": "Sufficient context collected"}


def route_label(state: MultiAgentState) -> Literal["research", "math", "finish"]:
    """
    Used by LangGraph conditional edge.
    """
    decision = decide_route(state)
    return decision["route_to"]
