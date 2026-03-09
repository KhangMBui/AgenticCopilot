"""
Retrieve tool - semantic search over knowledge base
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.tools.base import Tool, ToolParameter, ToolResult
from core.embeddings import OpenAIEmbeddingsClient
from app.models import Chunk, Document


class RetrieveTool(Tool):
    """
    Retrieve relevant information from the knowledge base.
    Uses vector similarity search.
    """

    def __init__(
        self, db: Session, embeddings_client: OpenAIEmbeddingsClient, workspace_id: int
    ):
        self.db = db
        self.embeddings_client = embeddings_client
        self.workspace_id = workspace_id

    def name(self) -> str:
        return "retrieve"

    def description(self) -> str:
        return "Search the knowledge base for relevant information. Use this when you need facts or context to answer"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="The search query to find relevant information",
                required=True,
            )
        ]

    def execute(self, query: str) -> ToolResult:
        """Execute semantic search."""
        try:
            # Embed query
            query_embedding = self.embeddings_client.embed_text(query)

            # Search
            score_expr = 1 - (Chunk.embedding.cosine_distance(query_embedding) / 2)

            stmt = (
                select(Chunk, Document.filename, score_expr.label("score"))
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.workspace_id == self.workspace_id)
                .where(Chunk.embedding.is_not(None))
                .where(score_expr >= 0.7)
                .order_by(score_expr.desc())
                .limit(3)
            )

            results = self.db.execute(stmt).all()

            if not results:
                return ToolResult(
                    success=True,
                    output="No relevant information found in knowledge base.",
                )

            # Format results
            output_parts = []
            for i, (chunk, filename, score) in enumerate(results, 1):
                output_parts.append(
                    f"[{i}] {filename} (relevance: {score:.2f})\n{chunk.content}\n"
                )
            return ToolResult(success=True, output="\n".join(output_parts))

        except Exception as e:
            return ToolResult(
                success=False, output=None, error=f"Retrieval failed : {str(e)}"
            )
