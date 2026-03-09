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
    retrieve_tool = RetrieveTool(db, embeddings, workspace_id)

    def research_worker(state: MultiAgentState) -> dict:
        step = state.get("step_count", 0) + 1
        query = state["query"]

        try:
            result = retrieve_tool.execute(query=query)
        except Exception as e:
            err = f"Retrieve exception: {type(e).__name__}: {e}"
            return {
                "worker_outputs": [
                    {
                        "worker_name": "research",
                        "success": False,
                        "content": "",
                        "sources": [],
                        "error": err,
                        "step": step,
                    }
                ],
                "trace": [
                    {
                        "step": step,
                        "node": "research_worker",
                        "success": False,
                        "error": err,
                    }
                ],
                "step_count": step,
            }
        text = (result.output or "").strip() if isinstance(result.output, str) else ""

        no_hit = not text or "No relevant information found" in text

        if result.success and not no_hit:
            content = result.output or ""
            note = content[
                :1200
            ]  # Keep memory bounded, what does this mean? It truncates saved context to 1200 chars so graph state does not grow too large across loops
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
                        "result_success": result.success,
                    }
                ],
                "step_count": step,
            }

        err = (
            (result.error or "").strip()
            or (text if text else "")
            or "Retrieve failed: no error details from tool"
        )
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
                    "result_success": result.success,
                    "result_output_preview": text[:200],
                }
            ],
            "step_count": step,
        }

    return research_worker
