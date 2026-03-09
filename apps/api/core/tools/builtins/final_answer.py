"""
Final answer tool - return answer to user and end agent loop.
"""

from core.tools.base import Tool, ToolParameter, ToolResult


class FinalAnswerTool(Tool):
    """
    Return final answer to the user.
    Use this when you have enough information to answer the question.
    """

    def name(self) -> str:
        return "final_answer"

    def description(self) -> str:
        return "Provide the final answer to the user's question. Use this when you have gathered enough information and are ready to respond."

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="answer",
                type="string",
                description="The complete answer to the user's question",
                required=True,
            )
        ]

    def execute(self, answer: str) -> ToolResult:
        """Return final answer."""
        return ToolResult(
            success=True,
            output=answer,
        )
