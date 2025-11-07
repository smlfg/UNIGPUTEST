"""
Vector Store using FAISS
Efficient similarity search for dense vectors
"""

from typing import List, Tuple, Optional
import numpy as np
import faiss
import pickle
from pathlib import Path


class VectorStore:
    """
    FAISS-based vector store for semantic search

    Supports:
    - Exact search (IndexFlatL2/IndexFlatIP)
    - Approximate search (IndexIVFFlat)
    - GPU acceleration (if available)
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        metric: str = "cosine",
        use_gpu: bool = False
    ):
        """
        Initialize vector store

        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ('flat', 'ivf')
            metric: Distance metric ('cosine', 'l2')
            use_gpu: Use GPU for search
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.use_gpu = use_gpu

        # Create FAISS index
        if index_type == "flat":
            if metric == "cosine":
                # For cosine similarity, use Inner Product with normalized vectors
                self.index = faiss.IndexFlatIP(dimension)
            else:
                # L2 distance
                self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            # IVF index for faster approximate search
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # GPU support
        if use_gpu and faiss.get_num_gpus() > 0:
            print(f"🚀 Using GPU for FAISS")
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        elif use_gpu:
            print("⚠️  GPU requested but not available, using CPU")

        # Storage for metadata
        self.texts: List[str] = []
        self.metadata: List[dict] = []
        self.doc_ids: List[str] = []

        print(f"✅ Vector store initialized ({index_type}, {metric}, dim={dimension})")

    def add_texts(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[dict] = None,
        doc_ids: List[str] = None
    ):
        """
        Add texts with their embeddings to the store

        Args:
            texts: List of text chunks
            embeddings: Embedding vectors (N x dimension)
            metadata: Optional metadata for each text
            doc_ids: Optional document IDs
        """
        if embeddings.shape[0] != len(texts):
            raise ValueError("Number of embeddings must match number of texts")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} != {self.dimension}")

        # Normalize embeddings for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)

        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store texts and metadata
        self.texts.extend(texts)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))

        if doc_ids:
            self.doc_ids.extend(doc_ids)
        else:
            self.doc_ids.extend([''] * len(texts))

        print(f"✅ Added {len(texts)} vectors (total: {self.index.ntotal})")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Tuple[List[str], List[float], List[dict]]:
        """
        Search for similar vectors

        Args:
            query_embedding: Query vector (dimension,)
            k: Number of results to return

        Returns:
            Tuple of (texts, scores, metadata)
        """
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query dimension {query_embedding.shape[0]} != {self.dimension}")

        # Reshape and normalize
        query_vector = query_embedding.reshape(1, -1).astype('float32')

        if self.metric == "cosine":
            faiss.normalize_L2(query_vector)

        # Search
        scores, indices = self.index.search(query_vector, k)

        # Extract results
        results_texts = []
        results_scores = []
        results_metadata = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results_texts.append(self.texts[idx])
                results_scores.append(float(score))
                results_metadata.append(self.metadata[idx])

        return results_texts, results_scores, results_metadata

    def save(self, save_path: str):
        """
        Save vector store to disk

        Args:
            save_path: Directory to save to
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(save_dir / "index.faiss"))

        # Save metadata
        metadata_dict = {
            'texts': self.texts,
            'metadata': self.metadata,
            'doc_ids': self.doc_ids,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
        }

        with open(save_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata_dict, f)

        print(f"✅ Vector store saved to {save_path}")

    @classmethod
    def load(cls, load_path: str, use_gpu: bool = False) -> 'VectorStore':
        """
        Load vector store from disk

        Args:
            load_path: Directory to load from
            use_gpu: Use GPU for search

        Returns:
            VectorStore instance
        """
        load_dir = Path(load_path)

        if not load_dir.exists():
            raise FileNotFoundError(f"Vector store not found: {load_path}")

        # Load metadata
        with open(load_dir / "metadata.pkl", 'rb') as f:
            metadata_dict = pickle.load(f)

        # Create instance
        store = cls(
            dimension=metadata_dict['dimension'],
            index_type=metadata_dict['index_type'],
            metric=metadata_dict['metric'],
            use_gpu=use_gpu
        )

        # Load FAISS index
        store.index = faiss.read_index(str(load_dir / "index.faiss"))

        if use_gpu and faiss.get_num_gpus() > 0:
            store.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, store.index)

        # Restore metadata
        store.texts = metadata_dict['texts']
        store.metadata = metadata_dict['metadata']
        store.doc_ids = metadata_dict['doc_ids']

        print(f"✅ Vector store loaded from {load_path} ({store.index.ntotal} vectors)")

        return store

    def get_stats(self) -> dict:
        """Get vector store statistics"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'total_texts': len(self.texts),
        }
