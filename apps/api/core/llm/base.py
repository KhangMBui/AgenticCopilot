"""
Base LLM client interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class Message:
    """Single message in conversation"""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    """LLM generation response."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient(ABC):
    """
    Abstract base for LLM providers.
    Allows swapping between OpenAI, Anthropic, local models, etc.
    """

    @abstractmethod
    def generate(
        self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 1000
    ) -> LLMResponse:
        """
        Generate completion from messages.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0-2)
            max_tokens: Max tokens to generate

        Returns:
            LLMResponse with content and usage
        """
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier."""
        pass
