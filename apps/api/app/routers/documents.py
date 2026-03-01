"""
Document ingestion endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.db import get_db
from app.models import Document, Chunk, Workspace
from app.schemas.documents import (
    DocumentCreateRequest,
    DocumentResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    ChunkResponse,
    DocumentListQuery,
)
from core.chunking import chunk_text
from core.embeddings import OpenAIEmbeddingsClient
from app.settings import settings

router = APIRouter(prefix="/workspaces/{workspace_id}/docs", tags=["documents"])


def _get_workspace_or_404(workspace_id: int, db: Session) -> Workspace:
    """Helper to get workspace or raise 404."""
    workspace = db.get(Workspace, workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worksapce {workspace_id} not found",
        )
    return workspace


@router.post(
    "", response_model=DocumentDetailResponse, status_code=status.HTTP_201_CREATED
)
def create_document(
    workspace_id: int,
    request: DocumentCreateRequest,
    db: Session = Depends(get_db),
):
    """
    Upload a document and chunk it.

    Steps:
    1. Validate workspace exists
    2. Create document record
    3. Chunk the content
    4. Store chunks
    5. Return document with chunks
    """

    # Validate workspace
    _get_workspace_or_404(workspace_id, db)

    # Create document
    doc = Document(
        workspace_id=workspace_id,
        filename=request.filename,
        content=request.content,
        mime_type=request.mime_type,
        size_bytes=len(request.content.encode("utf-8")),
    )
    db.add(doc)
    db.flush()  # Get doc.id

    # Chunk content
    chunk_results = chunk_text(text=request.content, chunk_size=1000, chunk_overlap=200)

    # Generate embeddings for all chunks
    embeddings_client = OpenAIEmbeddingsClient(api_key=settings.openai_api_key)
    texts = [cr.content for cr in chunk_results]
    embeddings = embeddings_client.embed_batch(texts)

    # Create chunk records with embeddings:
    for chunk_result, embedding in zip(chunk_results, embeddings):
        chunk = Chunk(
            document_id=doc.id,
            content=chunk_result.content,
            chunk_index=chunk_result.chunk_index,
            start_char=chunk_result.start_char,
            end_char=chunk_result.end_char,
            embedding=embedding,
        )
        db.add(chunk)

    db.commit()
    db.refresh(doc)

    # Build response
    return DocumentDetailResponse(
        id=doc.id,
        workspace_id=doc.workspace_id,
        filename=doc.filename,
        mime_type=doc.mime_type,
        size_bytes=doc.size_bytes,
        chunk_count=len(chunk_results),
        created_at=doc.created_at,
        chunks=[
            ChunkResponse(
                id=c.id,
                content=c.content,
                chunk_index=c.chunk_index,
                start_char=c.start_char,
                end_char=c.end_char,
                created_at=c.created_at,
            )
            for c in doc.chunks
        ],
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    workspace_id: int,
    query: DocumentListQuery = Depends(),
    db: Session = Depends(get_db),
):
    """List all documents in a workspace."""
    _get_workspace_or_404(workspace_id, db)

    # Count total
    total_stmt = select(func.count(Document.id)).where(
        Document.workspace_id == workspace_id
    )
    total = db.scalar(total_stmt) or 0

    # Get documents with chunk count
    stmt = (
        select(Document, func.count(Chunk.id).label("chunk_count"))
        .outerjoin(Chunk, Document.id == Chunk.document_id)
        .where(Document.workspace_id == workspace_id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
        .limit(query.limit)
        .offset(query.offset)
    )

    results = db.execute(stmt).all()

    documents = [
        DocumentResponse(
            id=doc.id,
            workspace_id=doc.workspace_id,
            filename=doc.filename,
            mime_type=doc.mime_type,
            size_bytes=doc.size_bytes,
            chunk_count=chunk_count,
            created_at=doc.created_at,
        )
        for doc, chunk_count in results
    ]

    return DocumentListResponse(
        documents=documents,
        total=total,
        limit=query.limit,
        offset=query.offset,
    )


@router.get("/{doc_id}", response_model=DocumentDetailResponse)
def get_document(
    workspace_id: int,
    doc_id: int,
    db: Session = Depends(get_db),
):
    """Get document details with all chunks."""
    _get_workspace_or_404(workspace_id, db)

    doc = db.get(Document, doc_id)
    if not doc or doc.workspace_id != workspace_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found in workspace {workspace_id}",
        )

    return DocumentDetailResponse(
        id=doc.id,
        workspace_id=doc.workspace_id,
        filename=doc.filename,
        mime_type=doc.mime_type,
        size_bytes=doc.size_bytes,
        chunk_count=len(doc.chunks),
        created_at=doc.created_at,
        chunks=[
            ChunkResponse(
                id=c.id,
                content=c.content,
                chunk_index=c.chunk_index,
                start_char=c.start_char,
                end_char=c.end_char,
                created_at=c.created_at,
            )
            for c in doc.chunks
        ],
    )


@router.get("/{doc_id}/chunks", response_model=list[ChunkResponse])
def get_document_chunks(
    workspace_id: int,
    doc_id: int,
    db: Session = Depends(get_db),
):
    """Get all chunks for a document."""
    _get_workspace_or_404(workspace_id, db)

    doc = db.get(Document, doc_id)
    if not doc or doc.workspace_id != workspace_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {doc_id} not found"
        )

    return [
        ChunkResponse(
            id=c.id,
            content=c.content,
            chunk_index=c.chunk_index,
            start_char=c.start_char,
            end_char=c.end_char,
            created_at=c.created_at,
        )
        for c in doc.chunks
    ]
