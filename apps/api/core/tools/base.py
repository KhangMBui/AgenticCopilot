"""
Base tool interface for ReAct agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolParameter:
  """Single tool parameter definition"""
  name: str
  type: str # "string", "number", "boolean"
  description: str
  required: bool = True

@dataclass
class ToolSchema:
  """Tool schema for validation and LLM prompting."""
  name: str
  description: str
  parameters: list[ToolParameter]

@dataclass
class ToolResult:
  """Result of tool execution"""
  success: bool
  output: Any # Tool-specific output
  error: str | None = None 


class Tool(ABC):
  """
  Abstract base class for agent tools.

  A tool is a capability the agent can use:
  - retrieve(query) -> search knowledge base
  - calculate(expression) -> evaluate math
  - web_search(query) -> search the web
  - final_answer(text) -> return answer to user
  """

  @abstractmethod
  def name(self) -> str:
    """Unique tool identifier."""
    pass

  @abstractmethod
  def description(self) -> str:
    """Human-readable description for the LLM"""
    pass

  @abstractmethod
  def parameters(self) -> list[ToolParameter]:
    """Tool parameters schema."""
    pass

  @abstractmethod
  def execute(self, **kwargs) -> ToolResult:
    """
    Execute the tool with given parameters.

    Args:
        **kwargs: Tool-specific parameters
    
    Returns:
        ToolResult with success status and output
    """
    pass

  def schema(self) -> ToolSchema:
    """Get tool schema for LLM prompting."""
    return ToolSchema(
      name=self.name(),
      description=self.description(),
      parameters=self.parameters(),
    )
  
  def validate_params(self, params: dict) -> tuple[bool, str | None]:
    """
    Validate parameters before execution.

    Returns:
        (is_valid, error_message)
    """
    schema = self.schema()

    # Check required params
    for param in schema.parameters:
      if param.required and param.name not in params:
        return False, f"Missing required parameter: {param.name}"
    
    # Check unexpected params
    valid_names = {p.name for p in schema.parameters}
    for key in params:
      if key not in valid_names:
        return False, f"Unexpected parameter: {key}"
    
    return True, None