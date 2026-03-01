"""
Base embeddings client interface.
"""

from abc import ABC, abstractmethod  # ABC = Abstract base class


class EmbeddingsClient(ABC):
    """
    Abstract base for embedding providers.
    Allows swapping OpenAI, Cohere, HuggingFace, etc.
    """

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embeddings for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats (embedding vector)
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (batch).

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension size."""
        pass
