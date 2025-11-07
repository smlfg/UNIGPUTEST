"""
Embedding Model
Uses Sentence Transformers for semantic embeddings
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingModel:
    """
    Embedding model using Sentence Transformers

    Default: all-MiniLM-L6-v2
    - 384 dimensions
    - Fast inference
    - Good quality
    - ~80MB model size
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None
    ):
        """
        Initialize embedding model

        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu/auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.device = device
        self.dimension = self.model.get_sentence_embedding_dimension()

        print(f"✅ Embedding model loaded (dimension: {self.dimension})")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        return embeddings

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query

        Args:
            query: Query text
            normalize: Normalize embedding

        Returns:
            Query embedding
        """
        return self.encode(query, normalize=normalize)[0]

    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode multiple documents

        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Document embeddings
        """
        return self.encode(
            documents,
            batch_size=batch_size,
            show_progress=show_progress
        )

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        return float(np.dot(embedding1, embedding2))

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'device': self.device,
            'max_seq_length': self.model.max_seq_length,
        }
