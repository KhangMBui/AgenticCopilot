"""
Embeddings clients
"""

from core.embeddings.base import EmbeddingsClient
from core.embeddings.openai_client import OpenAIEmbeddingsClient

# from core.embeddings.huggingface_client import HuggingFaceEmbeddingsClient

__all__ = [
    "EmbeddingsClient",
    "OpenAIEmbeddingsClient",
    #  "HuggingFaceEmbeddingsClient"
]
