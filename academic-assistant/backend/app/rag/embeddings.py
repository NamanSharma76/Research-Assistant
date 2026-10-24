"""Embedding model wrapper using sentence-transformers."""
from typing import List, Union
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME


class EmbeddingWrapper:
    """Singleton wrapper for sentence-transformers embedding model."""

    _instance: "EmbeddingWrapper | None" = None
    _model: SentenceTransformer | None = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL_NAME) -> "EmbeddingWrapper":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_name = model_name
            cls._instance._model = SentenceTransformer(model_name)
        return cls._instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of documents.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors as floats.
        """
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute embedding for a single search query.

        Args:
            text: Query text string.

        Returns:
            Embedding vector as float list.
        """
        embedding = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0].tolist()


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> EmbeddingWrapper:
    """Factory function returning the singleton embedding wrapper."""
    return EmbeddingWrapper(model_name)
