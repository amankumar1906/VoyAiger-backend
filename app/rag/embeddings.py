"""
Embedding generation module for RAG system.

This module handles text-to-vector conversion using SentenceTransformer models.
Uses all-mpnet-base-v2 (768 dimensions) for high-quality semantic embeddings.
"""

import logging
from typing import List, Optional
from functools import lru_cache

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for SentenceTransformer embedding model with caching.

    Features:
    - Lazy loading (model loaded on first use)
    - Singleton pattern (one model instance per process)
    - Batch processing support
    - Automatic normalization for cosine similarity
    """

    _instance: Optional['EmbeddingModel'] = None
    _model: Optional[SentenceTransformer] = None

    # Model configuration
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIMENSION = 768

    def __new__(cls):
        """Singleton pattern - only one instance per process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize embedding model (lazy loaded)."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.MODEL_NAME}")
            try:
                # Load model with CPU/GPU auto-detection
                self._model = SentenceTransformer(self.MODEL_NAME)
                logger.info(f"Embedding model loaded successfully. Dimension: {self.EMBEDDING_DIMENSION}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise RuntimeError(f"Could not load embedding model: {e}") from e

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of 768 float values (normalized for cosine similarity)

        Raises:
            ValueError: If text is empty
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        try:
            # Generate embedding (automatically normalized by the model)
            embedding = self._model.encode(
                text,
                normalize_embeddings=True,  # Normalize for cosine similarity
                show_progress_bar=False
            )

            # Convert numpy array to Python list
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once

        Returns:
            List of embeddings (each is a list of 768 floats)

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
            # Batch encode for efficiency
            embeddings = self._model.encode(
                valid_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Convert to list of lists
            return [emb.tolist() for emb in embeddings]

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
    return model.embed_text(text)


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


def get_embedding_model() -> EmbeddingModel:
    """
    Get global embedding model instance.

    Returns:
        Singleton EmbeddingModel instance
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
