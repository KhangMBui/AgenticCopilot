"""
LangChain tool wrappers for existing tools.
"""

from typing import Annotated
from langchain_core.tools import tool
from sqlalchemy.orm import Session

from core.embeddings import OpenAIEmbeddingsClient
from core.tools.builtins import RetrieveTool, CalculatorTool


def create_retrieve_tool(
    db: Session, embeddings: OpenAIEmbeddingsClient, workspace_id: int
):
    """
    Create a LangChain-compatible retrieve tool

    This wraps the existing manual RetrieveTool logic
    """

    # Create the existing tool instance
    retrieve_tool_instance = RetrieveTool(db, embeddings, workspace_id)

    # Wrap it as a LangChain tool
    @tool
    def retrieve(
        query: Annotated[str, "The search query to find relevant information"],
    ) -> str:
        """
        Search the knowledge base for relevant information
        Use this when need facts or context to answer the user's question
        """
        result = retrieve_tool_instance.execute(query=query)

        if result.success:
            return result.output
        else:
            return f"Error: {result.error}"

    return retrieve


def create_calculator_tool(
    db: Session, embeddings: OpenAIEmbeddingsClient, workspace_id: int
):
    """
    Create a LangChain-compatible calculator tool.
    """
    calculator_tool_instance = CalculatorTool()

    @tool
    def calculate(
        expression: Annotated[str, "Mathematical expression to calculate"],
    ) -> str:
        """
        Evaluate a mathematical expression.
        Use this for calculations like '2 + 2', '(10 * 5) / 2', etc.
        """
        result = calculator_tool_instance.execute(expression)

        if result.success:
            return str(result.output)
        else:
            return f"Error: {result.error}"

    return calculate
