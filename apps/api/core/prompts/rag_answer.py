"""
RAG answer generation prompt.
"""


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Build RAG prompt with retrieved context.

    Args:
        query: User's question
        context_chunks: List of dicts with keys:
            - content: chunk text
            - document_filename: source filename
            - chunk_index: position in document
            - score: relevance score

    Returns:
        Formatted prompt string
    """

    # Format context with citations
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[{i}] Source: {chunk['document_filename']} (chunk {chunk['chunk_index']})\n"
            f"{chunk['content']}\n"
        )

    context_text = "\n".join(context_parts)

    prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

# Context

{context_text}

# Instructions

1. Answer the question using ONLY information from the context above
2. Cite sources using [1], [2], etc. to reference the context chunks
3. If the context doesn't contain enough information, say so clearly
4. Be concise but complete
5. Use natural language - don't just copy/paste chunks

# Question

{query}

# Answer
"""

    return prompt


SYSTEM_PROMPT = """You are an AI assistant helping users find information in their knowledge base.

Your responses should:
- Be accurate and grounded in the provided context
- Include clear citations [1], [2] to source material
- Admit when information is not available in the context
- Be conversational and helpful

Do not:
- Make up information not present in the context
- Provide answers without citations
- Be overly verbose - keep responses focused
"""
