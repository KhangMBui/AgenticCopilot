"""
Backfill embeddings for existing chunks.

What this script does:
- Finds all Chunk rows in the database where embedding is NULL
- Calls an embeddings model (OpenAI) to generate a vector for each chunk's text
- Stores the vector back into Chunk.embedding
- Commits updates in batches to be efficient and resilient

Usage:
  docker compose exec api python scripts/embed_chunks.py
"""

import sys
from pathlib import Path

# When running a script directly (python scripts/...), Python might not treat your repo
# as a package, so imports like "from app.models import Chunk" may fail.
# This line adds the repo root folder to sys.path so imports work reliably.
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from app.db import SessionLocal
from app.models import Chunk
from core.embeddings import OpenAIEmbeddingsClient
from app.settings import settings


def embed_all_chunks():
    """
    Embed all chunks that don't have embeddings yet.

    High-level flow:
      1) Connect to DB
      2) SELECT chunks WHERE embedding IS NULL
      3) For those chunks: call embeddings API in batches
      4) Write embedding vectors back to DB
      5) Commit after each batch
    """

    # Create embeddings client (OpenAI). This object knows how to turn text -> vector.
    embeddings_client = OpenAIEmbeddingsClient(api_key=settings.openai_api_key)

    # Create a database session (connection wrapper).
    db = SessionLocal()
    try:
        # Build a SQL query:
        #   SELECT * FROM chunks WHERE embedding IS NULL;
        # NOTE: SQLAlchemy-style best practice is: Chunk.embedding.is_(None)
        stmt = select(Chunk).where(Chunk.embedding == None)  # noqa: E711

        # Execute query and load all matching Chunk objects into memory.
        chunks = db.scalars(stmt).all()

        # If there is nothing to backfill, exit early.
        if not chunks:
            print("All chunks already have embeddings")
            return

        print(f"📊 Found {len(chunks)} chunks to embed")

        # Batch size controls:
        # - how many chunk texts we send per API call
        # - memory usage
        # - rate-limit friendliness
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Extract the text content of each chunk for embedding.
            texts = [chunk.content for chunk in batch]

            print(f"🔄 Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")

            # This calls OpenAI embeddings endpoint once with a list of inputs.
            # It returns one vector per input text, typically 1536 floats each
            # when using text-embedding-3-small.
            embeddings = embeddings_client.embed_batch(texts)

            # Assign each returned vector to the matching chunk row.
            # zip() pairs:
            #   (batch[0], embeddings[0]), (batch[1], embeddings[1]), ...
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding

            # Commit changes for this batch so progress is saved even if later batches fail.
            db.commit()
            print(f"✅ Batch {i // batch_size + 1} complete")

        print(f"🎉 Successfully embedded {len(chunks)} chunks")

    finally:
        # Always close the DB session (even if error occurs).
        db.close()


if __name__ == "__main__":
    embed_all_chunks()
