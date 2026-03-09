"""
Parse tool calls from LLM output.

Why we need this:
- The LLM returns text.
- Our agent needs structured data:
    - tool_name (which tool to run)
    - params (JSON dict to pass into tool)
- So we parse the LLM output using regex and JSON parsing.

Expected LLM output format:

Thought: ...
Action: tool_name
Action Input: {"param": "value"}
"""

import re
import json
from typing import Tuple


def parse_tool_call(text: str) -> Tuple[str | None, dict | None, str | None]:
    """
    Parse tool call from LLM output.

    Expected format:
        Action: tool_name
        Action Input: {"param": "value"}

    Returns:
        (tool_name, params_dict, error_message)

    Examples:
      Input:
        "Thought: I should search\nAction: web_search\nAction Input: {\"q\": \"pgvector\"}"
      Output:
        ("web_search", {"q": "pgvector"}, None)

      Input:
        "Thought: Done\nAction: final_answer\nAction Input: {\"answer\": \"...\"}"
      Output:
        ("final_answer", {"answer": "..."}, None)
    """

    # ---------------------------------------------------------------------
    # 1) Extract Action line
    # ---------------------------------------------------------------------
    # Regex looks for: "Action:" followed by any characters until newline/end.
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if not action_match:
        # If we can't find Action, we can't proceed.
        return None, None, "No 'Action:' found in output"

    # tool_name is whatever comes after "Action:"
    tool_name = action_match.group(1).strip()

    # ---------------------------------------------------------------------
    # 2) Extract Action Input
    # ---------------------------------------------------------------------
    # We allow the input to be:
    # - JSON object: {...}
    # - JSON array:  [...]
    # - OR plain text fallback
    #
    # DOTALL allows '.' to match newlines (multi-line JSON).
    input_match = re.search(
        r"Action Input:\s*(\{.+?\})(?=\s*(?:\n|Observation:|$))",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    # If there is no Action Input line, default to empty params.
    # (Some tools might not require params.)
    if not input_match:
        return tool_name, {}, None

    input_str = input_match.group(1).strip()

    # ---------------------------------------------------------------------
    # 3) Try JSON parse
    # ---------------------------------------------------------------------
    # Tools expect params as dict, so JSON object is the best format.
    try:
        params = json.loads(input_str)
        # If user provided valid JSON but it's not an object (dict), that's not what we want.
        # Example: Action Input: ["a", "b"]
        if not isinstance(params, dict):
            return (
                tool_name,
                {},
                f"Action Input must be a JSON object, got {type(params)}",
            )
        return tool_name, params, None
    except json.JSONDecodeError:
        # -----------------------------------------------------------------
        # 4) Fallback: treat Action Input as plain string
        # -----------------------------------------------------------------
        # This lets the agent still call tools even if it didn't format JSON correctly.
        # Example: Action Input: hello world
        return tool_name, {"input": input_str}, None


def extract_thought(text: str) -> str:
    """
    Extract the Thought portion from the LLM output.

    We look for:
      Thought: ... until the next "Action" (or end of string)

    Why store thought?
    - Debugging
    - Agent trace UI
    - Understanding the agent's reasoning / plan
    """
    thought_match = re.search(
        r"Thought:\s*(.+?)(?:\n(?:Action|$))", text, re.IGNORECASE | re.DOTALL
    )
    if thought_match:
        return thought_match.group(1).strip()

    # If the model didn't follow format, fallback to returning the entire output
    # (so you still see what it said).
    return text.strip()
