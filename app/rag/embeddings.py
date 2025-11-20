"""
Embedding generation module for RAG system using Cohere API.

This module handles text-to-vector conversion using Cohere's embedding models.
Uses embed-english-v3.0 (1024 dimensions) - free tier available.
"""

import logging
from typing import List, Optional
from functools import lru_cache
import cohere

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for Cohere embedding API.

    Features:
    - API-based embeddings (no heavy model downloads)
    - Batch processing support
    - Free tier: 100 requests/minute
    - 1024-dimensional embeddings
    """

    _instance: Optional['EmbeddingModel'] = None
    _client: Optional[cohere.Client] = None

    # Model configuration
    MODEL_NAME = "embed-english-v3.0"
    EMBEDDING_DIMENSION = 1024
    INPUT_TYPE = "search_document"  # For indexing
    QUERY_INPUT_TYPE = "search_query"  # For retrieval

    def __new__(cls):
        """Singleton pattern - only one instance per process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Cohere client (loads API key from settings)."""
        if self._client is None:
            from ..config import settings
            api_key = settings.cohere_api_key

            if not api_key:
                raise ValueError("Cohere API key is required. Set COHERE_API_KEY in environment.")

            logger.info(f"Initializing Cohere client with model: {self.MODEL_NAME}")
            try:
                # Cohere v5 uses positional argument, not keyword
                self._client = cohere.Client(api_key)
                logger.info(f"Cohere client initialized. Dimension: {self.EMBEDDING_DIMENSION}")
            except Exception as e:
                logger.error(f"Failed to initialize Cohere client: {e}")
                raise RuntimeError(f"Could not initialize Cohere client: {e}") from e

    def embed_text(self, text: str, input_type: str = "search_document") -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed
            input_type: "search_document" for indexing, "search_query" for retrieval

        Returns:
            List of 1024 float values

        Raises:
            ValueError: If text is empty
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        try:
            # Call Cohere API
            response = self._client.embed(
                texts=[text],
                model=self.MODEL_NAME,
                input_type=input_type,
                embedding_types=["float"]
            )

            # Extract embedding
            embedding = response.embeddings.float[0]
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def embed_batch(self, texts: List[str], input_type: str = "search_document", batch_size: int = 96) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts
            input_type: "search_document" for indexing, "search_query" for retrieval
            batch_size: Number of texts to process at once (Cohere max: 96)

        Returns:
            List of embeddings (each is a list of 1024 floats)

        Raises:
            ValueError: If texts list is empty
            RuntimeError: If batch embedding fails
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")

        try:
            all_embeddings = []

            # Process in batches (Cohere limit: 96 texts per request)
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]

                response = self._client.embed(
                    texts=batch,
                    model=self.MODEL_NAME,
                    input_type=input_type,
                    embedding_types=["float"]
                )

                all_embeddings.extend(response.embeddings.float)

            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}") from e

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.EMBEDDING_DIMENSION

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self.MODEL_NAME


@lru_cache(maxsize=128)
def get_cached_embedding(text: str) -> List[float]:
    """
    Get embedding with caching for frequently used queries.

    Useful for:
    - User preference embeddings that don't change often
    - Common city/location queries
    - Repeated search queries

    Args:
        text: Input text to embed

    Returns:
        Cached or newly generated embedding

    Note:
        Cache size limited to 128 entries to prevent memory issues.
        Only cache immutable strings.
    """
    model = EmbeddingModel()
    return model.embed_text(text, input_type="search_query")


def create_document_text(
    city: str,
    preferences: str,
    itinerary_summary: str,
    feedback_text: Optional[str] = None
) -> str:
    """
    Create searchable document text from itinerary components.

    Combines multiple fields into a single text optimized for semantic search.

    Args:
        city: Destination city
        preferences: User preferences/interests
        itinerary_summary: High-level summary of activities
        feedback_text: Optional user feedback

    Returns:
        Formatted document text for embedding

    Example:
        >>> create_document_text(
        ...     "Paris",
        ...     "museums, French cuisine, romantic",
        ...     "Visited Louvre, Eiffel Tower, ate at Le Cinq",
        ...     "Amazing cultural experience!"
        ... )
        "Trip to Paris. Interests: museums, French cuisine, romantic.
         Activities: Visited Louvre, Eiffel Tower, ate at Le Cinq.
         Feedback: Amazing cultural experience!"
    """
    parts = [f"Trip to {city}."]

    if preferences and preferences.strip():
        parts.append(f"Interests: {preferences.strip()}.")

    if itinerary_summary and itinerary_summary.strip():
        parts.append(f"Activities: {itinerary_summary.strip()}.")

    if feedback_text and feedback_text.strip():
        parts.append(f"Feedback: {feedback_text.strip()}.")

    return " ".join(parts)


def create_query_text(city: str, preferences: str) -> str:
    """
    Create query text for retrieving similar past trips.

    Args:
        city: Destination city for new trip
        preferences: User preferences/interests for new trip

    Returns:
        Formatted query text for embedding

    Example:
        >>> create_query_text("Tokyo", "temples, sushi, cherry blossoms")
        "Planning trip to Tokyo with interests: temples, sushi, cherry blossoms"
    """
    query = f"Planning trip to {city}"

    if preferences and preferences.strip():
        query += f" with interests: {preferences.strip()}"

    return query


# Global singleton instance
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(api_key: Optional[str] = None) -> EmbeddingModel:
    """
    Get global embedding model instance.

    Args:
        api_key: Optional Cohere API key (ignored, uses settings)

    Returns:
        Singleton EmbeddingModel instance
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
