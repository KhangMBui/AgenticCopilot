"""
M6 Phase 3: Multi-agent graph with supervisor routing

What this file is responsible for:
- Build the LangGraph multi-agent workflow
- Register the supervisor + worker nodes
- Define how routing happens between nodes
- Run the graph for one user query

Big picture:
START -> supervisor -> (research OR math OR finish)
research -> supervisor
math -> supervisor
finish -> END
"""

# Lets us use newer type-hint behavior more cleanly.
# Helps with forward references in type hints.
from __future__ import annotations

# Literal means a function can only return specific exact string values.
# Example:
# Literal["research", "math", "finish"]
# means ONLY those 3 strings are valid return values.
from typing import Literal

# AIMessage is the message object used in LangChain / LangGraph conversation history.
from langchain_core.messages import AIMessage

# START and END are special graph markers.
# StateGraph is the graph object where we register nodes and edges.
from langgraph.graph import START, END, StateGraph

# SQLAlchemy DB session type. The research worker needs DB access.
from sqlalchemy.orm import Session

# Shared state schema for the whole graph.
# This is the "big dictionary" that all nodes read from and write to.
from core.agents.langgraph_multi.state import MultiAgentState

# Helper function that creates the starting state before graph execution.
from core.agents.langgraph_multi.multi_agent import build_initial_state

# Function that contains the supervisor's routing logic.
# It should decide whether to send the task to research, math, or finish.
from core.agents.langgraph_multi.supervisor import decide_route

# Factory function that creates the research worker node.
from core.agents.langgraph_multi.research import create_research_worker

# Factory function that creates the math worker node.
from core.agents.langgraph_multi.math import create_math_worker

# Grounded final-answer aggregation for final node.
from core.agents.langgraph_multi.aggregator import compose_grounded_answer

# Embeddings client, likely used by the research worker for retrieval / semantic search.
from core.embeddings import OpenAIEmbeddingsClient


def create_supervisor_node():
    """
    Creates and returns the supervisor node function.

    Why this is a factory:
    - Sometimes node factories are used so you can configure the node before returning it.
    - In this specific version, it just returns the inner function.

    The supervisor node:
    - reads the current shared state
    - asks the routing logic what to do next
    - writes the decision into state as "next_node"
    - appends a supervisor message into conversation history
    - appends a trace/debug record
    """

    def supervisor_node(state: MultiAgentState) -> dict:
        """
        One execution of the supervisor node.

        Input:
        - state: the current shared graph state

        Output:
        - a partial state update dictionary
        """
        # Get route decision
        decision = decide_route(state)

        return {
            # Save the supervisor’s chosen next step into state.
            # This value will later be read by route_from_supervisor().
            "next_node": decision["route_to"],
            # Save the explanation / reasoning for the route.
            "plan": decision["reason"],
            # Append a human-readable supervisor message into state.messages.
            # This helps with debugging and gives visibility into routing decisions.
            "messages": [
                AIMessage(
                    content=f"Supervisor: route -> {decision['route_to']} ({decision['reason']})"
                )
            ],
            # Append structured debug information to the trace.
            # Useful for observing graph execution later.
            "trace": [
                {
                    "node": "supervisor",
                    "step": state.get("step_count", 0),
                    "route_to": decision["route_to"],
                    "reason": decision["reason"],
                }
            ],
        }

    # Return the actual node function so it can be registered into the graph.
    return supervisor_node


def finish_node(state: MultiAgentState) -> dict:
    """
    Phase 4 finish node: grounded aggregation.
    """
    final = compose_grounded_answer(state)

    return {
        "final_answer": final,
        "messages": [AIMessage(content=final)],
        "trace": [
            {
                "node": "finish",
                "step": state.get("step_count", 0),
                "success": True,
                "grounded": True,
            }
        ],
    }


def route_from_supervisor(
    state: MultiAgentState,
) -> Literal["research", "math", "finish"]:
    """
    Routing function used by LangGraph after the supervisor runs.

    It reads state["next_node"] and tells LangGraph where to go next.

    Returns only one of:
    - "research"
    - "math"
    - "finish"
    """
    return state.get("next_node", "finish")


def create_multi_agent_graph(
    db: Session,
    embeddings: OpenAIEmbeddingsClient,
    workspace_id: int,
):
    """
    Builds and compiles the full multi-agent graph.

    Inputs:
    - db: database session for research access
    - embeddings: embeddings client for semantic retrieval
    - workspace_id: identifies which workspace's data to use

    Steps:
    1. Create node functions
    2. Create the StateGraph
    3. Register nodes
    4. Register edges
    5. Compile the graph into an executable app
    """

    # Create the supervisor node function
    supervisor = create_supervisor_node()

    # Create the research worker node, configured with DB + embeddings + workspace
    research = create_research_worker(
        db=db, embeddings=embeddings, workspace_id=workspace_id
    )

    # Create the math worker node
    math = create_math_worker()

    # Build a graph whose shared state structure is MultiAgentState
    graph = StateGraph(MultiAgentState)

    # Register all nodes by name
    graph.add_node("supervisor", supervisor)
    graph.add_node("research", research)
    graph.add_node("math", math)
    graph.add_node("finish", finish_node)

    # Graph always begins at supervisor
    graph.add_edge(START, "supervisor")

    # After supervisor runs:
    # - call route_from_supervisor(state)
    # - depending on the returned string,
    #   move to the matching node
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"research": "research", "math": "math", "finish": "finish"},
    )

    # After research finishes, go back to supervisor
    graph.add_edge("research", "supervisor")

    # After math finishes, go back to supervisor
    graph.add_edge("math", "supervisor")

    # After finish node runs, the graph ends
    graph.add_edge("finish", END)

    # Compile into an executable LangGraph app
    return graph.compile()


def run_multi_agent(
    query: str,
    db: Session,
    embeddings: OpenAIEmbeddingsClient,
    workspace_id: int,
    max_steps: int = 8,
) -> dict:
    """
    High-level entry point for running the multi-agent system.

    What it does:
    1. Build the graph
    2. Build the initial state
    3. Invoke the graph
    4. Return a cleaner summarized result dictionary
    """

    # Build the compiled LangGraph app
    app = create_multi_agent_graph(
        db=db, embeddings=embeddings, workspace_id=workspace_id
    )

    # Create the starting graph state
    # Usually includes the user query, empty worker buckets, counters, etc.
    state = build_initial_state(query=query, max_steps=max_steps)

    # Run the graph until it reaches END
    result = app.invoke(state)

    return {
        "query": query,  # Original user query
        # Consider run successful if a final_answer exists
        "status": "completed" if result.get("final_answer") else "failed",
        # Final answer produced by the graph
        "final_answer": result.get("final_answer"),
        # Number of steps taken, if tracked in state
        "step_count": result.get("step_count", 0),
        # Debug trace across nodes
        "trace": result.get("trace", []),
        # Structured worker outputs collected during execution
        "worker_outputs": result.get("worker_outputs", []),
    }
