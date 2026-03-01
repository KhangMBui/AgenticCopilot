"""
OpenAI embeddings client.
"""

from openai import OpenAI
from core.embeddings.base import EmbeddingsClient


class OpenAIEmbeddingsClient(EmbeddingsClient):
    """
    OpenAI embeddings using text-embedding-3-small.
    - 1536 dimensions
    - ~$0.02 per 1M tokens
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536  # text-embedding-3-small

    def embed_text(self, text: str) -> list[float]:
        """Embed single text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one API call."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension
