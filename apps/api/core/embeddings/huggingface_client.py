# """
# HuggingFace embeddings client.
# """

# from sentence_transformers import SentenceTransformer
# from core.embeddings.base import EmbeddingsClient


# class HuggingFaceEmbeddingsClient(EmbeddingsClient):
#     """
#     Local HuggingFace embeddings using sentence-transformers.
#     """

#     def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
#         self.model = SentenceTransformer(model_name)
#         self._dimension = self.model.get_sentence_embedding_dimension()

#     def embed_text(self, text: str) -> list[float]:
#         """Embed single text."""
#         embedding = self.model.encode(text, normalize_embeddings=True)
#         return embedding.tolist()

#     def embed_batch(self, texts: list[str]) -> list[list[float]]:
#         """Embed multiple texts in one API call."""
#         embeddings = self.model.encode(texts, normalize_embeddings=True)
#         return [emb.tolist() for emb in embeddings]

#     def dimension(self) -> int:
#         """Return embedding dimension."""
#         return self._dimension
