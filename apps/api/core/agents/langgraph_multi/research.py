"""
M6 Phase 2: Research worker node
"""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage
from sqlalchemy.orm import Session

from core.agents.langgraph_multi.state import MultiAgentState, WorkerOutput
from core.embeddings import OpenAIEmbeddingsClient
from core.tools.builtins import RetrieveTool


def create_research_worker(
    db: Session, embeddings: OpenAIEmbeddingsClient, workspace_id: int
) -> Callable[[MultiAgentState], dict]:
    """
    Factory returning a LangGraph-compatible node function
    """
    retrieve_tool = RetrieveTool(
        db=db, embeddings_client=embeddings, workspace_id=workspace_id
    )

    def research_worker(state: MultiAgentState) -> dict:
        step = state.get("step_count", 0) + 1
        query = state["query"]

        result = retrieve_tool.execute(query=query)

        if result.success:
            content = result.output or ""
            note = content[:1200]  # Keep memory bounded, what does this mean?
            worker_output: WorkerOutput = {
                "worker_name": "research",
                "success": True,
                "content": note,
                "sources": [],  # optional: parse sources from retrieve output later
                "error": None,
                "step": step,
            }

            return {
                "research_notes": [note],
                "worker_outputs": [worker_output],
                "messages": [
                    AIMessage(content="Research worker retrieved relevant context.")
                ],
                "trace": [
                    {
                        "step": step,
                        "node": "research_worker",
                        "success": True,
                        "query": query,
                    }
                ],
                "step_count": step,
            }

        err = result.error or "Unknown retrieve error"
        worker_output = {
            "worker_name": "research",
            "success": False,
            "content": "",
            "sources": [],
            "error": err,
            "step": step,
        }
        return {
            "worker_outputs": [worker_output],
            "messages": [AIMessage(content=f"Research worker failed: {err}")],
            "trace": [
                {
                    "step": step,
                    "node": "research_worker",
                    "success": False,
                    "query": query,
                    "error": err,
                }
            ],
            "step_count": step,
            "error": err,
        }

    return research_worker
