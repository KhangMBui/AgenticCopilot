"""
M6 Phase 4: grounded final-answer aggregation

Purpose of this file:
---------------------
This helper builds the final answer returned by the multi-agent system.

Instead of just dumping worker outputs, it:
1. Organizes results into sections
2. Shows retrieved context and calculations
3. Adds grounding metadata (worker success/failure)

This improves transparency and makes the answer easier to debug and trust.
"""

# Enables forward references in type hints.
from __future__ import annotations

# Shared state structure used across the entire LangGraph workflow.
from core.agents.langgraph_multi.state import MultiAgentState


# Helper function for finish_node()
def compose_grounded_answer(state: MultiAgentState) -> str:
    """
    Builds a grounded final answer from the multi-agent state.

    Input:
        state: MultiAgentState
            The shared state object that contains:
            - query
            - research_notes
            - math_results
            - worker_outputs

    Output:
        str
            A formatted final answer string combining worker outputs.
    """

    # Original user query.
    query = state["query"]

    # Context retrieved by research worker(s).
    # Usually text chunks, summarized knowledge, etc.
    research_notes = state.get("research_notes", [])

    # Results produced by math worker(s).
    # Usually calculations or derived numeric outputs.
    math_results = state.get("math_results", [])

    # Structured outputs from each worker execution.
    # Typically contains fields like:
    #   worker_name
    #   success
    #   error
    #   metadata
    worker_outputs = state.get("worker_outputs", [])

    # ---------------------------------------------------------
    # Build final answer line-by-line
    # ---------------------------------------------------------

    # We'll assemble the answer incrementally as a list of lines.
    lines: list[str] = []

    # Start the output with the user query.
    lines.append(f"Answer to {query}\n")

    # ---------------------------------------------------------
    # Section 1: Retrieved Context (Research Worker)
    # ---------------------------------------------------------
    if research_notes:
        # This section shows the evidence retrieved by the research agent.
        lines.append("## Retrieved Context")

        # Only include the most recent research note.
        # Using [-1] keeps the output concise if multiple retrieval passes occurred.
        # strip() removes leading/trailing whitespace.
        lines.append(research_notes[-1].strip())

    # ---------------------------------------------------------
    # Section 2: Math Calculations (Math Worker)
    # ---------------------------------------------------------
    if math_results:
        lines.append("\n## Calculations")

        # If multiple calculations occurred,
        # list them sequentially.
        for i, r in enumerate(math_results, 1):
            lines.append(f"{i}. {r}")
            # Example output:
            # 1, Revenue growth = 25%
            # 2, Total profit = $42,000

    # ---------------------------------------------------------
    # Section 3: Grounding / Confidence Metadata
    # ---------------------------------------------------------
    # This section provides transparency about what workers succeeded or failed.
    lines.append("\n## Grounding Notes")

    # Workers that successfully completed their tasks.
    ok_workers = [w for w in worker_outputs if w.get("success")]

    # Workers that failed
    bad_workers = [w for w in worker_outputs if not w.get("success")]

    # Shows how many workers completed successfully.
    lines.append(f"- Successful workers: {len(ok_workers)}")

    # If there are bad workers, show which worker failed and why.
    # Example:
    # - math_worker: division by zero
    # - research_worker: timeout
    if bad_workers:
        lines.append(f"- Failed workers: {len(bad_workers)}")
        for w in bad_workers:
            lines.append(
                f"  - {w.get('worker_name')}: {w.get('error') or 'unknown error'}"
            )

    # ---------------------------------------------------------
    # Fallback case
    # ---------------------------------------------------------
    if not research_notes and not math_results:
        lines.append("- No usable worker output collected")

    # ---------------------------------------------------------
    # Final formatted string
    # ---------------------------------------------------------
    # Join all lines into a single formatted answer string.
    return "\n".join(lines).strip()
