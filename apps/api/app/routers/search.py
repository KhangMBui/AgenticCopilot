"""
Semantic search endpoint using pgvector.

What this endpoint does:
- User sends a query string (e.g., "How do I reset my password?")
- We convert that query into a vector embedding (list of floats)
- We compare that query vector to every chunk's stored embedding in Postgres
- We return the chunk whose vectors are "closest" (most semantically similar)

This is semantic search:
- Not keyword match ("password" appears)
- But meaning match ("forgot login", "reset credentials", etc.)
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.db import get_db
from app.models import Chunk, Document
from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from core.embeddings import OpenAIEmbeddingsClient
from app.settings import settings

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def semantic_search(
    request: SearchRequest,
    db: Session = Depends(get_db),
):
    """
    Semantic search using vector similarity (cosine distance).

    High-level steps:
    1. Convert the user's query text into an embedding vector (query_embedding)
    2. Ask Postgres/pgvector: "Which chunk embeddings are closest to query_bemdding?"
    3. Return the top N chunks, ranked by closeness (similarity score)

    Note:
    - This is teh retrieval part of RAG.
    - Later, the chat endpoint will take these retrieved chunks and feed them into the LLM
    """

    # ------------------------------------------------------------------------
    # STEP 1) Embed the user query (text -> vector)
    # ------------------------------------------------------------------------
    embeddings_client = OpenAIEmbeddingsClient(api_key=settings.openai_api_key)

    # Generate a vector for the query text
    query_embedding = embeddings_client.embed_text(request.query)

    # ------------------------------------------------------------------------
    # STEP 2) Vector similarity serach in Postgres (pgvector)
    # ------------------------------------------------------------------------
    # In our database:
    # - each chunk row stores:
    #     content: the chunk text
    #     embedding: the vector representation of that text
    #
    # We want:
    #     "Return the chunks whose embedding vectors are closest to query_embedding"
    #
    # pgvector supports distance functions (cosine, L2, inner product).
    # Here, we use cosine distance:
    #
    #     cosine_distance(a, b) = 1 - cosine_similarity(a, b)
    #
    # cosine_similarity ranges [-1, 1]
    # cosine_distance ranges [0, 2] if vectors are not normalized (depending on implementation)
    #
    # We're converting distance into a "score" where higher is better:
    # score = 1 - (distance /2)
    #
    # So:
    # - distance close to 0 -> score close to 1 (very similar)
    # - distance close to 2 -> score close to 0 (very dissimilar)

    # Build the SQL query using SQLAlchemy.
    #
    # The query selects:
    #   - Chunk (the whole ORM object)
    #   - Document.filename (so we can show which doc it came from)
    #   - A computed "score" based on cosine distance
    #
    # NOTE: order_by_distance ascending means closest first
    stmt = (
        # SELECT Chunk, Document.filename, (1 - cosine_distance / 2) AS score
        # JOIN Document
        # ON Chunk.document_id = Document.id
        # ORDER BY ASC(cosine_distance)
        # LIMIT request.limit
        select(
            Chunk,
            Document.filename,
            # Convert cosine distance to similarity score
            # distance in [0, 2] -> score in [1, 0]
            # 1 - (distance / 2)
            (1 - (Chunk.embedding.cosine_distance(query_embedding) / 2)).label("score"),
        )
        # Join Document so we can access metadata like filename/workspace_id
        .join(Document, Chunk.document_id == Document.id)
        # Ranking:
        # We want the nearest vectors first, so we order by distance ASC
        # (smaller distance = more similar).
        .order_by((Chunk.embedding.cosine_distance(query_embedding)).asc())
        # Limit number of results returned
        .limit(request.limit)
    )

    # Optional filter (if provided):
    # If workspace_id is provided, only search within that workspace's document.
    # This is super important for multi-tenant apps (separating customer data).
    if request.workspace_id:
        stmt = stmt.where(Document.workspace_id == request.workspace_id)

    # Execute query.
    # Result is a list of tuples:
    #   (Chunk, filename, score)
    results = db.execute(stmt).all()

    # ------------------------------------------------------------------------
    # STEP 3) Convert DB results into response object
    # ------------------------------------------------------------------------
    # We build API response models (Pydantic).
    # The endpoint returns:
    # - the original query
    # - a list of top matching chunks
    # - each result includes score + metadata
    search_results = [
        SearchResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_filename=filename,
            content=chunk.content,
            score=float(score),
            chunk_index=chunk.chunk_index,
        )
        for chunk, filename, score in results
    ]

    return SearchResponse(
        query=request.query,
        results=search_results,
        total_results=len(search_results),
    )
