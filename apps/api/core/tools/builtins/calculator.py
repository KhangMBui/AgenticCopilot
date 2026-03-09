"""
Calculator tool - evaluate math expression
"""

import ast  # abstract syntax tree
import operator

from core.tools.base import Tool, ToolParameter, ToolResult


class CalculatorTool(Tool):
    """
    Safely evaluate mathematical expressions.
    Supports: +, -, *, /, **, (), basic functions.
    """

    # Allowed operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def name(self) -> str:
        return "calculate"

    def description(self) -> str:
        return "Evaluate a mathematical expression. Use this for calculations like '2 + 2', '(10 * 5) / 2', etc."

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate (e.g., '2 + 2', '(10 * 5) / 2')",
                required=True,
            )
        ]

    def execute(self, expression: str) -> ToolResult:
        """Safely evaluate math expression."""
        try:
            # Parse expression
            tree = ast.parse(expression, mode="eval")

            # Evaluate
            result = self._eval_node(tree.body)

            return ToolResult(
                success=True,
                output=str(result),
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid expression: {str(e)}",
            )

    def _eval_node(self, node):
        """Recursively evaluate AST node."""
        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.BinOp):
            op = self.OPERATORS.get(type(node.op))
            if not op:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            op = self.OPERATORS.get(type(node.op))
            if not op:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            operand = self._eval_node(node.operand)
            return op(operand)

        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")
