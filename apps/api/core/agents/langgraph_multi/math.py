"""
M6 Phase 2: Math worker node.
"""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.messages import AIMessage

from core.agents.langgraph_multi.state import MultiAgentState, WorkerOutput
from core.tools.builtins import CalculatorTool


def _extract_expression(query: str) -> str | None:
    """
    Best-effort extraction of a math expression from user query.
    Example: 'what is (5 * 100) / 2?' -> '(5 * 100) / 2'
    """
    # keep digits/operators/parens/space
    candidate = "".join(ch for ch in query if ch.isdigit() or ch in "+-*/().% ")
    candidate = re.sub(r"\s+", " ", candidate).strip()

    # require at least one digit and one operator
    if re.search(r"\d", candidate) and re.search(r"[+\-*/%]", candidate):
        return candidate
    return None


def create_math_worker() -> Callable[[MultiAgentState], dict]:
    """
    Factory returning a LangGraph-compatible node function.
    """
    calculator = CalculatorTool()

    def math_worker(state: MultiAgentState) -> dict:
        step = state.get("step_count", 0) + 1
        query = state["query"]

        expression = _extract_expression(query)
        if not expression:
            msg = "No valid math expression found in query."
            worker_output: WorkerOutput = {
                "worker_name": "math",
                "success": False,
                "content": "",
                "sources": [],
                "error": msg,
                "step": step,
            }
            return {
                "worker_outputs": [worker_output],
                "messages": [AIMessage(content=f"Math worker skipped: {msg}")],
                "trace": [
                    {
                        "step": step,
                        "node": "math_worker",
                        "success": False,
                        "error": msg,
                    }
                ],
                "step_count": step,
            }

        result = calculator.execute(expression=expression)

        if result.success:
            answer = str(result.output)
            worker_output = {
                "worker_name": "math",
                "success": True,
                "content": answer,
                "sources": [],
                "error": None,
                "step": step,
            }
            return {
                "math_results": [answer],
                "worker_outputs": [worker_output],
                "messages": [AIMessage(content=f"Math worker result: {answer}")],
                "trace": [
                    {
                        "step": step,
                        "node": "math_worker",
                        "success": True,
                        "expression": expression,
                        "result": answer,
                    }
                ],
                "step_count": step,
            }

        err = result.error or "Unknown calculation error"
        worker_output = {
            "worker_name": "math",
            "success": False,
            "content": "",
            "sources": [],
            "error": err,
            "step": step,
        }
        return {
            "worker_outputs": [worker_output],
            "messages": [AIMessage(content=f"Math worker failed: {err}")],
            "trace": [
                {
                    "step": step,
                    "node": "math_worker",
                    "success": False,
                    "expression": expression,
                    "error": err,
                }
            ],
            "step_count": step,
            "error": err,
        }

    return math_worker
