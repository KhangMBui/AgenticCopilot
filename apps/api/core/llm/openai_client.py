"""
OpenAI LLM Client.
"""

from openai import OpenAI
from core.llm.base import LLMClient, Message, LLMResponse


class OpenAIClient(LLMClient):
    """
    OpenAI chat completion client.
    Default: gpt-40-mini (fast, cheap, good for RAG)
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 1000
    ) -> LLMResponse:
        """Generate completion."""

        # Convert to openAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # Call API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract response
        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content,
            model=response.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def model_name(self) -> str:
        """Return model name."""
        return self.model
