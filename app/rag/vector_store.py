"""
Vector store operations for RAG system using Supabase pgvector.

This module handles CRUD operations for travel_documents table:
- Inserting embeddings with metadata
- Similarity search using cosine distance
- Deleting outdated/low-rated documents
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID

from app.utils.database import SupabaseClient

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Interface to Supabase pgvector for storing and retrieving travel document embeddings.
    """

    TABLE_NAME = "travel_documents"

    def __init__(self):
        """Initialize vector store with Supabase client."""
        self.supabase = SupabaseClient.get_client()

    async def insert_document(
        self,
        user_id: UUID,
        itinerary_id: UUID,
        feedback_id: UUID,
        document_text: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Insert a new travel document with embedding.

        Args:
            user_id: User who owns this document
            itinerary_id: Associated itinerary ID
            feedback_id: Associated feedback ID (unique)
            document_text: Searchable text content
            embedding: 768-dimensional vector
            metadata: Additional context (city, dates, rating, etc.)

        Returns:
            Inserted document data or None if failed

        Raises:
            Exception: If insertion fails
        """
        try:
            # Prepare document data
            document = {
                "user_id": str(user_id),
                "itinerary_id": str(itinerary_id),
                "feedback_id": str(feedback_id),
                "document_text": document_text,
                "embedding": embedding,
                "metadata": metadata,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            # Insert into Supabase
            response = self.supabase.table(self.TABLE_NAME).insert(document).execute()

            if response.data and len(response.data) > 0:
                logger.info(f"Inserted document for feedback_id={feedback_id}, user_id={user_id}")
                return response.data[0]
            else:
                logger.warning(f"Insert returned no data for feedback_id={feedback_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to insert document for feedback_id={feedback_id}: {e}")
            raise

    async def upsert_document(
        self,
        user_id: UUID,
        itinerary_id: UUID,
        feedback_id: UUID,
        document_text: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Upsert (insert or update) a travel document.

        Uses feedback_id as unique key. If document exists, updates it.
        Useful when user updates their feedback rating.

        Args:
            user_id: User who owns this document
            itinerary_id: Associated itinerary ID
            feedback_id: Associated feedback ID (unique constraint)
            document_text: Searchable text content
            embedding: 768-dimensional vector
            metadata: Additional context

        Returns:
            Upserted document data or None if failed
        """
        try:
            # Check if document already exists for this feedback
            existing = await self.get_document_by_feedback_id(feedback_id)

            if existing:
                # Update existing document
                return await self.update_document(
                    document_id=UUID(existing["id"]),
                    document_text=document_text,
                    embedding=embedding,
                    metadata=metadata
                )
            else:
                # Insert new document
                return await self.insert_document(
                    user_id=user_id,
                    itinerary_id=itinerary_id,
                    feedback_id=feedback_id,
                    document_text=document_text,
                    embedding=embedding,
                    metadata=metadata
                )

        except Exception as e:
            logger.error(f"Failed to upsert document for feedback_id={feedback_id}: {e}")
            raise

    async def update_document(
        self,
        document_id: UUID,
        document_text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing document.

        Args:
            document_id: ID of document to update
            document_text: New document text (optional)
            embedding: New embedding vector (optional)
            metadata: New metadata (optional)

        Returns:
            Updated document data or None if failed
        """
        try:
            # Build update payload
            update_data = {"updated_at": datetime.utcnow().isoformat()}

            if document_text is not None:
                update_data["document_text"] = document_text
            if embedding is not None:
                update_data["embedding"] = embedding
            if metadata is not None:
                update_data["metadata"] = metadata

            # Update in Supabase
            response = self.supabase.table(self.TABLE_NAME)\
                .update(update_data)\
                .eq("id", str(document_id))\
                .execute()

            if response.data and len(response.data) > 0:
                logger.info(f"Updated document {document_id}")
                return response.data[0]
            else:
                logger.warning(f"Update returned no data for document_id={document_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            raise

    async def delete_document_by_feedback_id(self, feedback_id: UUID) -> bool:
        """
        Delete a document by its feedback_id.

        Used when:
        - User updates feedback rating to < 4 stars
        - User deletes their feedback
        - Itinerary is deleted (CASCADE should handle this)

        Args:
            feedback_id: Feedback ID to delete document for

        Returns:
            True if deleted, False if not found
        """
        try:
            response = self.supabase.table(self.TABLE_NAME)\
                .delete()\
                .eq("feedback_id", str(feedback_id))\
                .execute()

            if response.data and len(response.data) > 0:
                logger.info(f"Deleted document for feedback_id={feedback_id}")
                return True
            else:
                logger.info(f"No document found for feedback_id={feedback_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete document for feedback_id={feedback_id}: {e}")
            raise

    async def get_document_by_feedback_id(self, feedback_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by feedback_id.

        Args:
            feedback_id: Feedback ID to look up

        Returns:
            Document data or None if not found
        """
        try:
            response = self.supabase.table(self.TABLE_NAME)\
                .select("*")\
                .eq("feedback_id", str(feedback_id))\
                .execute()

            if response.data and len(response.data) > 0:
                return response.data[0]
            return None

        except Exception as e:
            logger.error(f"Failed to get document by feedback_id={feedback_id}: {e}")
            return None

    async def similarity_search(
        self,
        query_embedding: List[float],
        user_id: UUID,
        limit: int = 5,
        filter_city: Optional[str] = None,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using Supabase RPC function.

        Args:
            query_embedding: 768-dimensional query vector
            user_id: User ID to filter by (privacy)
            limit: Maximum number of results
            filter_city: Optional city filter (e.g., "Paris")
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of matching documents with similarity scores, sorted by relevance

        Note:
            Uses RPC function 'match_travel_documents' created in migration
        """
        try:
            # Call Supabase RPC function
            response = self.supabase.rpc(
                "match_travel_documents",
                {
                    "query_embedding": query_embedding,
                    "match_user_id": str(user_id),
                    "match_count": limit,
                    "filter_city": filter_city
                }
            ).execute()

            if not response.data:
                logger.info(f"No similar documents found for user_id={user_id}")
                return []

            # Filter by minimum similarity
            results = [
                doc for doc in response.data
                if doc.get("similarity", 0) >= min_similarity
            ]

            logger.info(
                f"Found {len(results)} similar documents for user_id={user_id} "
                f"(min_similarity={min_similarity})"
            )

            return results

        except Exception as e:
            logger.error(f"Similarity search failed for user_id={user_id}: {e}")
            # Don't raise - return empty list to allow graceful degradation
            return []

    async def get_user_document_count(self, user_id: UUID) -> int:
        """
        Get count of indexed documents for a user.

        Args:
            user_id: User ID

        Returns:
            Number of documents indexed for this user
        """
        try:
            response = self.supabase.table(self.TABLE_NAME)\
                .select("id", count="exact")\
                .eq("user_id", str(user_id))\
                .execute()

            return response.count or 0

        except Exception as e:
            logger.error(f"Failed to get document count for user_id={user_id}: {e}")
            return 0

    async def delete_user_documents(self, user_id: UUID) -> int:
        """
        Delete all documents for a user.

        Used when user deletes their account.

        Args:
            user_id: User ID

        Returns:
            Number of documents deleted
        """
        try:
            response = self.supabase.table(self.TABLE_NAME)\
                .delete()\
                .eq("user_id", str(user_id))\
                .execute()

            deleted_count = len(response.data) if response.data else 0
            logger.info(f"Deleted {deleted_count} documents for user_id={user_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete documents for user_id={user_id}: {e}")
            raise


# Global singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get global VectorStore instance.

    Returns:
        Singleton VectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
