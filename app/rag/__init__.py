"""
RAG (Retrieval-Augmented Generation) module for VoyAiger.

This module provides personalized itinerary generation by learning from
user's past highly-rated trips (4-5 stars).

Main components:
- embeddings: Text-to-vector conversion using SentenceTransformer
- vector_store: Supabase pgvector operations (insert, search, delete)
- retriever: High-level RAG workflows (index, retrieve, format)

Usage:
    # Index a high-rated itinerary
    from app.rag import get_retriever
    retriever = get_retriever()
    await retriever.index_itinerary_feedback(
        user_id=user_id,
        itinerary_id=itinerary_id,
        feedback_id=feedback_id,
        city="Paris",
        preferences="museums, cuisine",
        rating=5
    )

    # Retrieve personalization context
    context = await retriever.get_personalization_context(
        user_id=user_id,
        city="Tokyo",
        preferences="temples, sushi"
    )
"""

from app.rag.embeddings import (
    EmbeddingModel,
    get_embedding_model,
    create_document_text,
    create_query_text
)

from app.rag.vector_store import (
    VectorStore,
    get_vector_store
)

from app.rag.retriever import (
    RAGRetriever,
    get_retriever
)

__all__ = [
    # Embeddings
    "EmbeddingModel",
    "get_embedding_model",
    "create_document_text",
    "create_query_text",

    # Vector Store
    "VectorStore",
    "get_vector_store",

    # Retriever (main interface)
    "RAGRetriever",
    "get_retriever",
]
