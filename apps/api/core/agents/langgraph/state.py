"""
LangGraph agent state definition
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    State for LangGraph ReAct agent

    LangGraph tracks state changes automatically.
    The 'messages' field uses a special reducer (add_messages)
    that appends new messages without overwriting
    """

    # Conversation history - LangGraph will manage this
    messages: Annotated[list[BaseMessage], add_messages]

    # Original Query
    query: str

    # Step counter (for max_steps safety)
    step_count: int

    # Final answer (when agent is done)
    final_answer: str | None
