"""
LangGraph ReAct agent implementation
"""

from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from core.agents.langgraph.state import AgentState
from core.agents.langgraph.tool_wrappers import (
    create_retrieve_tool,
    create_calculator_tool,
)
from sqlalchemy.orm import Session
from core.embeddings import OpenAIEmbeddingsClient


def create_langgraph_agent(
    api_key: str,
    db: Session,
    embeddings: OpenAIEmbeddingsClient,
    workspace_id: int,
    max_steps: int = 10,
):
    """
    Create a LangGraph ReAct agent.

    Args:
        api_key: OpenAI API key
        db: Database session
        embeddings: Embeddings client
        workspace_id: Workspace to search
        max_steps: maximum steps before stopping
    Return:
        Compiled LangGraph StateGraph
    """
    # ---------------------------------------------------------------------
    # 1. Create tools
    # ---------------------------------------------------------------------
    tools = [
        create_retrieve_tool(db, embeddings, workspace_id),
        create_calculator_tool(db, embeddings, workspace_id),
    ]

    # ---------------------------------------------------------------------
    # 2. Create LLM with tool binding
    # ---------------------------------------------------------------------
    # LangGraph automatically handles tool calling with OpenAI's function calling
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0,
    )

    # Bind tools to LLM - this enables native tool calling
    llm_with_tools = llm.bind_tools(tools)

    # ---------------------------------------------------------------------
    # 3. Define agent node (Think + Decide)
    # ---------------------------------------------------------------------
    def agent_node(state: AgentState) -> AgentState:
        """
        Agent reasoning node.

        This is where the LLM thinks and decides what to do next.
        """
        # Build system prompt
        system_prompt = """You are a helpful AI assistant with access to tools.

        Think step-by-step about how to answer the user's question.

        Use the retrieve tool to search the knowledge base for relevant information.
        Use the calculate tool for mathematical operations.

        When you have enough information, provide a final answer directly (don't call a tool).

        Be concise and cite sources when using retrieved information.
        """

        # Get conversation history
        messages = state["messages"]

        # Add system prompt if this is the first call
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages

        # Call LLM (with tool calling capability)
        response = llm_with_tools.invoke(messages)

        # Increment step counter
        step_count = state.get("step_count", 0) + 1

        return {
            "messages": [response],
            "step_count": step_count,
        }

    # ---------------------------------------------------------------------
    # 4. Define conditional edge logic
    # ---------------------------------------------------------------------
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """
        Decide whether to continue or end.

        End conditions:
        1. No tool calls in last message (agent gave final answer)
        2. Exceeded max steps
        """

        message = state["messages"]
        last_message = message[-1]
        step_count = state.get("step_count", 0)

        # Check max steps
        if step_count >= max_steps:
            return "end"

        # Check if LLM wants to call tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # No tool calls - agent is done
        return "end"

    # ---------------------------------------------------------------------
    # 5. Build the graph
    # ---------------------------------------------------------------------
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))  # LangGraph's built-in tool executor

    # Define edges'
    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # After tools execute, go back to agent
    workflow.add_edge("tools", "agent")

    # Compile the graph
    app = workflow.compile()

    return app


def run_langgraph_agent(
    query: str,
    api_key: str,
    db: Session,
    embeddings: OpenAIEmbeddingsClient,
    workspace_id: int,
    max_steps: int = 10,
) -> dict:
    """
    Run the LangGraph agent and extract results

    Args:
        query: User's question
        api_key: OpenAI API key
        db: Database session
        embeddings: Embeddings client
        workspace_id: Workspace to search
        max_steps: Maximum steps

    Returns:
        dict with final_answer, steps, status
    """

    # Create agent graph
    app = create_langgraph_agent(
        api_key=api_key,
        db=db,
        embeddings=embeddings,
        workspace_id=workspace_id,
        max_steps=max_steps,
    )

    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "step_count": 0,
        "final_answer": None,
    }

    # Run the graph (where does this invoke come from? It's an inherited function from LangChain's Runnable interface)
    # Since we did workflow.compile() -> returns CompiledStateGraph, which is inherited from Runnable (langchain_core.runnables.base.Runnable)
    # -> .invoke(input) is a function here
    result = app.invoke(initial_state)

    # Extract final answer
    messages = result["messages"]
    last_message = messages[-1]
    final_answer = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

    # Extract steps for debugging
    steps = []
    for i, msg in enumerate(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                steps.append(
                    {
                        "step": i,
                        "type": "tool_call",
                        "tool": tool_call["name"],
                        "args": tool_call["args"],
                    }
                )
        elif hasattr(msg, "content"):
            steps.append(
                {
                    "step": i,
                    "type": "message",
                    "content": msg.content[:200],  # Truncate for readability
                }
            )

    return {
        "final_answer": final_answer,
        "steps": steps,
        "status": "completed" if result["step_count"] < max_steps else "max_steps",
        "total_steps": result["step_count"],
    }
