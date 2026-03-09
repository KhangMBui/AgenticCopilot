"""
M6, phase 1: Multi-agent scaffold (contracts only)
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from core.agents.langgraph_multi.state import MultiAgentState


def build_initial_state(query: str, max_steps: int = 8) -> MultiAgentState:
    """
    Shared initial state used by the graph runner.
    """
    return MultiAgentState(
        messages=[HumanMessage(content=query)],
        query=query,
        plan=None,
        next_node="research",
        research_notes=[],
        math_results=[],
        worker_outputs=[],
        trace=[],
        step_count=0,
        max_steps=max_steps,
        final_answer=None,
        draft_answer=None,
        error=None,
    )
