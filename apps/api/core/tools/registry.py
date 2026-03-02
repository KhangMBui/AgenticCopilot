"""
Tool registry for agent system
"""

from typing import Dict
from core.tools.base import Tool, ToolSchema, ToolResult


class ToolRegistry:
    """
    Central registry for all available tools.

    Responsibilities:
    - Register tools dynamically
    - Validatre tool uniqueness
    - Provide tool schemas for LLM prompting
    - Execute tools by name
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        name = tool.name()
        if name in self._tools:
            raise ValueError(f"Tool {name} already registered")
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        """Get tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[ToolSchema]:
        """Get schemas for all tools (for LLM prompting)."""
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, tool_name: str, params: dict):
        """
        Execute a tool by name.

        Returns:
            ToolResult
        """
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False, output=None, error=f"Tool '{tool_name}' not found"
            )

        # Validate params
        is_valid, error = tool.validate_params(params)
        if not is_valid:
            return ToolResult(success=False, output=None, error=error)

        # Execute
        return tool.execute(**params)
