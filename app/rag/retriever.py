"""
High-level retrieval logic for RAG system.

This module provides user-friendly interfaces for:
- Indexing itineraries when feedback is submitted
- Retrieving relevant past trips for personalization
- Managing document lifecycle (update/delete)
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID

from app.rag.embeddings import (
    get_embedding_model,
    create_document_text,
    create_query_text
)
from app.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    High-level interface for RAG operations.

    Combines embedding generation and vector storage into simple workflows.
    """

    def __init__(self):
        """Initialize retriever with embedding model and vector store."""
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store()

    async def index_itinerary_feedback(
        self,
        user_id: UUID,
        itinerary_id: UUID,
        feedback_id: UUID,
        city: str,
        start_date: str,
        end_date: str,
        preferences: str,
        itinerary_summary: str,
        rating: int,
        feedback_text: Optional[str] = None
    ) -> bool:
        """
        Index an itinerary when user provides feedback ≥ 4 stars.

        This is the main entry point for adding documents to RAG.

        Args:
            user_id: User who created the itinerary
            itinerary_id: Itinerary ID
            feedback_id: Feedback ID (unique)
            city: Destination city
            start_date: Trip start date (ISO format)
            end_date: Trip end date (ISO format)
            preferences: User preferences/interests
            itinerary_summary: Summary of activities (from itinerary_data)
            rating: Feedback rating (1-5)
            feedback_text: Optional user feedback text

        Returns:
            True if successfully indexed, False otherwise

        Note:
            Only index if rating >= 4. Caller should check this before calling.
        """
        try:
            # Validate rating
            if rating < 4:
                logger.warning(
                    f"Attempted to index low-rated itinerary (rating={rating}). "
                    "Only 4-5 star itineraries should be indexed."
                )
                return False

            # Create searchable document text
            document_text = create_document_text(
                city=city,
                preferences=preferences,
                itinerary_summary=itinerary_summary,
                feedback_text=feedback_text
            )

            # Generate embedding
            logger.info(f"Generating embedding for feedback_id={feedback_id}")
            embedding = self.embedding_model.embed_text(document_text)

            # Prepare metadata
            metadata = {
                "city": city,
                "start_date": start_date,
                "end_date": end_date,
                "preferences": preferences,
                "rating": rating
            }

            # Upsert to vector store (handles updates if feedback changes)
            await self.vector_store.upsert_document(
                user_id=user_id,
                itinerary_id=itinerary_id,
                feedback_id=feedback_id,
                document_text=document_text,
                embedding=embedding,
                metadata=metadata
            )

            logger.info(
                f"Successfully indexed itinerary for user_id={user_id}, "
                f"city={city}, rating={rating}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to index itinerary feedback: {e}")
            return False

    async def remove_itinerary_feedback(self, feedback_id: UUID) -> bool:
        """
        Remove an itinerary from RAG index.

        Used when:
        - User updates feedback rating to < 4 stars
        - User deletes their feedback
        - Itinerary quality no longer meets threshold

        Args:
            feedback_id: Feedback ID to remove

        Returns:
            True if successfully removed, False otherwise
        """
        try:
            deleted = await self.vector_store.delete_document_by_feedback_id(feedback_id)
            if deleted:
                logger.info(f"Removed document for feedback_id={feedback_id} from RAG index")
            return deleted

        except Exception as e:
            logger.error(f"Failed to remove itinerary feedback: {e}")
            return False

    async def retrieve_similar_trips(
        self,
        user_id: UUID,
        city: str,
        preferences: str,
        limit: int = 3,
        min_similarity: float = 0.6,
        same_city_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve user's past trips similar to their current planning.

        This is the main entry point for RAG retrieval during itinerary generation.

        Args:
            user_id: User requesting personalization
            city: Destination city for new trip
            preferences: User preferences/interests for new trip
            limit: Maximum number of similar trips to return
            min_similarity: Minimum similarity threshold (0-1, default 0.6)
            same_city_only: If True, only return trips to the same city

        Returns:
            List of similar past trips with metadata and similarity scores

        Example return value:
            [
                {
                    "id": "uuid",
                    "document_text": "Trip to Paris. Interests: museums...",
                    "metadata": {"city": "Paris", "rating": 5, ...},
                    "similarity": 0.85,
                    "created_at": "2024-01-15T10:00:00"
                },
                ...
            ]
        """
        try:
            # Check if user has any indexed documents
            doc_count = await self.vector_store.get_user_document_count(user_id)
            if doc_count == 0:
                logger.info(f"User {user_id} has no indexed trips yet")
                return []

            # Create query text
            query_text = create_query_text(city=city, preferences=preferences)

            # Generate query embedding (use "search_query" input type for retrieval)
            query_embedding = self.embedding_model.embed_text(query_text, input_type="search_query")

            # Perform similarity search
            filter_city = city if same_city_only else None
            similar_docs = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                user_id=user_id,
                limit=limit,
                filter_city=filter_city,
                min_similarity=min_similarity
            )

            if similar_docs:
                logger.info(f"✓ RAG: Retrieved {len(similar_docs)} similar trips for {city}")
            else:
                logger.info(f"RAG: No similar trips found for {city}")

            return similar_docs

        except Exception as e:
            logger.error(f"Failed to retrieve similar trips: {e}")
            # Graceful degradation - return empty list
            return []

    async def format_rag_context(
        self,
        similar_trips: List[Dict[str, Any]],
        max_context_length: int = 1000
    ) -> str:
        """
        Format similar trips into LLM context string.

        Args:
            similar_trips: List of similar trips from retrieve_similar_trips()
            max_context_length: Maximum character length for context

        Returns:
            Formatted context string for LLM prompt injection

        Example output:
            "Based on your past trips:
            1. Paris (5⭐, 85% match): You enjoyed museums, French cuisine, and romantic settings.
            2. Rome (4⭐, 78% match): You loved historical sites and authentic Italian food."
        """
        if not similar_trips:
            return ""

        context_parts = ["Based on your past highly-rated trips:"]

        for idx, trip in enumerate(similar_trips, 1):
            metadata = trip.get("metadata", {})
            city = metadata.get("city", "Unknown")
            rating = metadata.get("rating", 0)
            preferences = metadata.get("preferences", "")
            similarity = trip.get("similarity", 0)

            # Format similarity as percentage
            similarity_pct = int(similarity * 100)

            # Create trip summary
            stars = "⭐" * rating
            trip_summary = f"{idx}. {city} ({stars}, {similarity_pct}% match)"

            if preferences:
                trip_summary += f": You enjoyed {preferences}."

            context_parts.append(trip_summary)

        # Join and truncate if needed
        context = "\n".join(context_parts)
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        return context

    async def get_personalization_context(
        self,
        user_id: UUID,
        city: str,
        preferences: str,
        limit: int = 3
    ) -> str:
        """
        One-shot method to get RAG context for itinerary generation.

        Combines retrieval and formatting into single call.

        Args:
            user_id: User requesting itinerary
            city: Destination city
            preferences: User preferences
            limit: Max number of past trips to consider

        Returns:
            Formatted context string (empty if no relevant trips)

        Usage in travel agent:
            rag_context = await get_personalization_context(user_id, city, preferences)
            if rag_context:
                prompt = f"{rag_context}\n\n{original_prompt}"
        """
        try:
            # Retrieve similar trips
            similar_trips = await self.retrieve_similar_trips(
                user_id=user_id,
                city=city,
                preferences=preferences,
                limit=limit,
                min_similarity=0.6,  # Moderate threshold
                same_city_only=False  # Learn from all destinations
            )

            # Format into context
            if similar_trips:
                return await self.format_rag_context(similar_trips)
            else:
                return ""

        except Exception as e:
            logger.error(f"Failed to get personalization context: {e}")
            # Graceful degradation
            return ""


# Global singleton instance
_retriever: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    """
    Get global RAGRetriever instance.

    Returns:
        Singleton RAGRetriever instance
    """
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
