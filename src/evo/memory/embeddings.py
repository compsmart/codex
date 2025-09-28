"""
Embedding generation and management for Evo AI memory system.
"""

import asyncio
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingManager:
    """Manages text embeddings for semantic memory search."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_size: int = 1000
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_size = cache_size
        self._model: Optional[SentenceTransformer] = None
        self._cache: Dict[str, List[float]] = {}
        self._embedding_dimension: Optional[int] = None

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None, self._load_model
        )

        # Determine embedding dimension
        test_embedding = await self.embed_text("test")
        self._embedding_dimension = len(test_embedding)

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        model = SentenceTransformer(self.model_name, device=self.device)
        return model

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            return [0.0] * (self._embedding_dimension or 384)

        # Check cache first
        if text in self._cache:
            return self._cache[text]

        if self._model is None:
            raise RuntimeError("Embedding model not initialized")

        # Generate embedding
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self._model.encode, text
        )

        # Convert to list and cache
        embedding_list = embedding.tolist()

        # Manage cache size
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[text] = embedding_list
        return embedding_list

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        if self._model is None:
            raise RuntimeError("Embedding model not initialized")

        # Check cache for all texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text in self._cache:
                embeddings.append(self._cache[text])
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            loop = asyncio.get_event_loop()
            new_embeddings = await loop.run_in_executor(
                None, self._model.encode, uncached_texts
            )

            # Update cache and results
            for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                embedding_list = new_embeddings[i].tolist()
                embeddings[idx] = embedding_list

                # Manage cache size
                if len(self._cache) >= self.cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

                self._cache[text] = embedding_list

        return embeddings

    async def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embedding1 = await self.embed_text(text1)
        embedding2 = await self.embed_text(text2)

        return self._cosine_similarity(
            np.array(embedding1),
            np.array(embedding2)
        )

    async def find_similar_texts(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[tuple[str, float]]:
        """Find the most similar texts to a query."""
        query_embedding = await self.embed_text(query)
        candidate_embeddings = await self.embed_texts(candidates)

        similarities = []
        query_vec = np.array(query_embedding)

        for text, embedding in zip(candidates, candidate_embeddings):
            if embedding:
                similarity = self._cosine_similarity(
                    query_vec,
                    np.array(embedding)
                )
                if similarity >= min_similarity:
                    similarities.append((text, similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self._embedding_dimension is None:
            # Default for all-MiniLM-L6-v2
            return 384
        return self._embedding_dimension

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the embedding cache."""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.embedding_dimension
        }