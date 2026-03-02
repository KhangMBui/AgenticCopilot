"""
RAG (Retrieval-Augmented Generation) chat endpoint.

What "RAG chat" means:
- A normal chatbot answers based on its pretrained knowledge (and may hallucinate).
- A RAG chatbot first "retrieves" relevant facts from your own knowledge base
  (your uploaded docs, chunks, etc.), then asks the LLM to answer using ONLY those facts.

This file implements:
- POST /chat : ask a question and get an answer + cited sources
- GET /chat/conversations : list conversations in a workspace
- GET /chat/conversations/{id}/messages : get messages and their citations

Core idea:
1) Embed the user message into a vector
2) Vector-search the database for similar chunk embeddings
3) Build a prompt that includes the retrieved chunks as "context"
4) Call the LLM to generate an answer grounded in that context
5) Store everything to DB so chats are persistent
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.db import get_db
from app.models import Conversation, Message, Chunk, Document, Workspace
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    MessageResponse,
    CitedSource,
    ConversationResponse,
)
from core.llm import OpenAIClient, Message as LLMMessage
from core.embeddings import OpenAIEmbeddingsClient
from core.prompts.rag_answer import build_rag_prompt, SYSTEM_PROMPT
from app.settings import settings

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_or_create_conversation(
    workspace_id: int,
    conversation_id: int | None,
    db: Session,
) -> Conversation:
    """
    Get an existing conversation if conversation_id is provided,
    otherwise create a new conversation.

    Why do this?
    - We want persistent chat history per workspace.
    - The frontend can continue the same conversation by passing conversation_id.
    """

    # ---------------------------------------------------------------------
    # Validate workspace exists
    # ---------------------------------------------------------------------
    # This prevents users from writing messages into a workspace that doesn't exist.
    workspace = db.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace {workspace_id} not found",
        )

    # ---------------------------------------------------------------------
    # If conversation_id is provided, load it and validate it belongs to a workspace
    # ---------------------------------------------------------------------
    if conversation_id:
        conv = db.get(Conversation, conversation_id)
        if not conv or conv.workspace_id != workspace_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found",
            )
        return conv

    # ---------------------------------------------------------------------
    # Otherwise create a new conversation
    # ---------------------------------------------------------------------
    conv = Conversation(workspace_id=workspace_id)
    db.add(conv)
    db.flush()
    return conv


def _retrieve_context(
    query: str, workspace_id: int, db: Session, limit: int = 5
) -> list[dict]:
    """
    Retrieve relevant chunks from our knowledge base using semantic vector search.

    Inputs:
    - query: the user's question/message
    - workspace_id: search only docs belonging to that workspace (multi-tenant safety)
    - limit: top-k results to return

    Output:
    - A list of dicts with chunk_id, content, doc filename, chunk index, score

    This is the "Retrieval" in RAG.
    """

    MIN_RELEVANCE_SCORE = 0.7

    # ---------------------------------------------------------------------
    # STEP 1) Embed the query text into a vector
    # ---------------------------------------------------------------------
    # The embedding model maps text -> vector (list of floats).
    # Similar meaning => vectors are close in vector space.
    embeddings_client = OpenAIEmbeddingsClient(api_key=settings.openai_api_key)
    query_embedding = embeddings_client.embed_text(query)

    # Calculate score as a subquery column
    score_expr = 1 - (Chunk.embedding.cosine_distance(query_embedding) / 2)

    # ---------------------------------------------------------------------
    # STEP 2) Vector search in Postgres using pgvector
    # ---------------------------------------------------------------------
    # We want the nearest chunk embeddings to the query embedding.
    # We use cosine distance:
    #   cosine_distance(a, b) = 1 - cosine_similarity(a, b)
    #
    # Smaller distance => more similar.
    stmt = (
        select(
            Chunk,
            Document.filename,
            # Compute similarity score from cosine distance.
            # This is mostly for presentation / debugging.
            score_expr.label("score"),
        )
        .join(Document, Chunk.document_id == Document.id)
        # Multi-tenant safety: only search docs within this workspace
        .where(Document.workspace_id == workspace_id)
        # IMPORTANT: avoid NULL embeddings (if backfill isn't complete)
        # If you have chunks with NULL embedding, distance ops can fail.
        .where(Chunk.embedding.is_not(None))
        .where(score_expr > MIN_RELEVANCE_SCORE)  # Filter out low relevance score
        # Rank by distance ascending (nearest neighbors first)
        .order_by((Chunk.embedding.cosine_distance(query_embedding)).asc())
        # Return only top-k
        .limit(limit)
    )

    # Execute query: returns tuples (Chunk, filename, score)
    results = db.execute(stmt).all()

    # Format results into a clean structure for prompt building + response
    return [
        {
            "chunk_id": chunk.id,
            "content": chunk.content,
            "document_filename": filename,
            "chunk_index": chunk.chunk_index,
            "score": float(score),
        }
        for chunk, filename, score in results
    ]


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    RAG chat endpoint.

    Flow:
    1) Get or create conversation (ensures chat history is persistent)
    2) Store the user's message in DB
    3) Retrieve relevant context chunks using semantic vector search (pgvector)
    4) Build a RAG prompt that includes the retrieved context
    5) Ask the LLM to answer using that prompt
    6) Store assistant answer + citations + token usage in DB
    7) Return the assistant answer + sources to the client
    """

    # ---------------------------------------------------------------------
    # 1) Get/create conversation
    # ---------------------------------------------------------------------
    conversation = _get_or_create_conversation(
        workspace_id=request.workspace_id,
        conversation_id=request.conversation_id,
        db=db,
    )

    # ---------------------------------------------------------------------
    # 2) Store user message
    # ---------------------------------------------------------------------
    # We store every message to DB so:
    # - chat can be reloaded later
    # - we can do analytics / evals
    # - we can add memory later
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
    )
    db.add(user_message)
    db.flush()  # assigns user_message.id without committing yet

    # ---------------------------------------------------------------------
    # 3) Retrieve context (top-k most relevant chunks)
    # ---------------------------------------------------------------------
    context_chunks = _retrieve_context(
        query=request.message,
        workspace_id=request.workspace_id,
        db=db,
        limit=5,  # k = 5
    )

    # We'll fill these either with a fallback response or LLM-generated response
    assistant_content: str
    cited_chunk_ids: list[int]

    # We'll only define llm_client/llm_response if we actually call the LLM
    llm_client = None
    llm_response = None

    # ---------------------------------------------------------------------
    # If no context is found, return a safe fallback message
    # ---------------------------------------------------------------------
    if not context_chunks:
        assistant_content = "I couldn't find any relevant information in the knowledge base to answer your question."
        cited_chunk_ids = []
    else:
        # -----------------------------------------------------------------
        # 4) Build RAG prompt
        # -----------------------------------------------------------------
        # build_rag_prompt typically formats:
        # - the question
        # - the retrieved chunks (with IDs / filenames)
        # - instructions to cite sources
        rag_prompt = build_rag_prompt(
            query=request.message, context_chunks=context_chunks
        )

        # -----------------------------------------------------------------
        # 5) Generate answer using LLM
        # -----------------------------------------------------------------
        llm_client = OpenAIClient(api_key=settings.openai_api_key)

        # We provide:
        # - a SYSTEM prompt that sets behavior rules
        # - a USER message containing the RAG prompt (question + context)
        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=rag_prompt),
        ]

        # Temperature controls randomness (0.0 = most deterministic).
        # max_tokens controls maximum response length.
        llm_response = llm_client.generate(messages, temperature=0.7, max_tokens=1000)

        assistant_content = llm_response.content

        # For now we cite all retrieved chunks.
        # Later improvements:
        # - ask model to return explicit citations, and parse which chunks were used
        cited_chunk_ids = [chunk["chunk_id"] for chunk in context_chunks]

    # ---------------------------------------------------------------------
    # 6) Store assistant message
    # ---------------------------------------------------------------------
    # Save assistant answer, plus:
    # - which model was used
    # - token usage/cost (useful for observability)
    # - which chunks were cited
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=assistant_content,
        model=llm_client.model_name() if llm_client else None,
        prompt_tokens=llm_response.prompt_tokens if llm_response else None,
        completion_tokens=llm_response.completion_tokens if llm_response else None,
        total_tokens=llm_response.total_tokens if llm_response else None,
        cited_chunk_ids=cited_chunk_ids,
    )
    db.add(assistant_message)

    # Optionally set conversation title from first user message
    if not conversation.title:
        # conversation.title = request.message[:100] # sliced from message

        # Generate concise title from first message using LLM prompting
        title_prompt = f"Generate a 5-word title for this question: {request.message}"
        title_response = llm_client.generate(
            [LLMMessage(role="user", content=title_prompt)],
            temperature=0.3,
            max_tokens=20,
        )
        conversation.title = title_response.content.strip('"')

    db.commit()
    db.refresh(assistant_message)

    # ---------------------------------------------------------------------
    # 7) Build API response: include cited sources + answer
    # ---------------------------------------------------------------------
    sources = [
        CitedSource(
            chunk_id=chunk["chunk_id"],
            document_filename=chunk["document_filename"],
            content=chunk["content"],
            chunk_index=chunk["chunk_index"],
            relevance_score=chunk["score"],
        )
        for chunk in context_chunks
    ]

    return ChatResponse(
        conversation_id=conversation.id,
        message=MessageResponse(
            id=assistant_message.id,
            role=assistant_message.role,
            content=assistant_message.content,
            created_at=assistant_message.created_at,
            model=assistant_message.model,
            total_tokens=assistant_message.total_tokens,
            cited_sources=sources,
        ),
        sources=sources,
    )


@router.get("/conversations", response_model=list[ConversationResponse])
def list_conversations(workspace_id: int, db: Session = Depends(get_db)):
    """
    List all conversations in a workspace.

    We return a message_count per conversation using:
    - OUTER JOIN messages
    - GROUP BY conversation
    - COUNT(message.id)

    This supports a frontend "conversation list" UI.
    """

    # Build a query to get all conversations in a workspace
    stmt = (
        select(Conversation, func.count(Message.id).label("message_count"))
        .outerjoin(Message, Conversation.id == Message.conversation_id)
        .where(Conversation.workspace_id == workspace_id)
        .group_by(Conversation.id)
        .order_by(Conversation.updated_at.desc())
    )

    # Execute the query
    results = db.execute(stmt).all()

    return [
        ConversationResponse(
            id=conv.id,
            workspace_id=conv.workspace_id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=msg_count,
        )
        for conv, msg_count in results
    ]


@router.get(
    "/conversations/{conversation_id}/messages", response_model=list[MessageResponse]
)
def get_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
):
    """
    Return all messages in a conversation + reconstruct cited sources.

    How citations are stored:
    - assistant messages store a list of cited_chunk_ids (e.g., [12, 99, 103])
    - when returning conversation history, we re-fetch those chunks and attach their content

    Note:
    - We do not store relevance_score per message, so we set 0.0.
      If we'd want that later, store scores in Message as well.
    """

    conversation = db.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    messages = conversation.messages

    # Build responses with sources
    response_messages = []
    for msg in messages:
        sources = []
        if msg.role == "assistant" and msg.cited_chunk_ids:
            # Fetch cited chunks
            chunks = db.query(Chunk).filter(Chunk.id.in_(msg.cited_chunk_ids)).all()
            chunk_map = {c.id: c for c in chunks}

            for chunk_id in msg.cited_chunk_ids:
                chunk = chunk_map.get(chunk_id)
                if chunk:
                    sources.append(
                        CitedSource(
                            chunk_id=chunk.id,
                            document_filename=chunk.document.filename,
                            content=chunk.content,
                            chunk_index=chunk.chunk_index,
                            relevance_score=0.0,  # Not stored historically in message
                        )
                    )
        response_messages.append(
            MessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at,
                model=msg.model,
                total_tokens=msg.total_tokens,
                cited_sources=sources,
            )
        )

    return response_messages
