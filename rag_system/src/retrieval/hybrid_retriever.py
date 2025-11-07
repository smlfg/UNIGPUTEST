"""
Hybrid Retriever
Combines semantic (FAISS) and keyword (BM25) search
"""

from typing import List, Tuple, Dict
import numpy as np
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Hybrid retrieval combining semantic and keyword search

    Workflow:
    1. Retrieve top-K results from FAISS (semantic)
    2. Retrieve top-K results from BM25 (keyword)
    3. Merge and re-rank results using weighted scores
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever

        Args:
            embedding_model: Embedding model for queries
            vector_store: FAISS vector store
            bm25_retriever: BM25 retriever
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        self.semantic_weight = semantic_weight / total_weight
        self.keyword_weight = keyword_weight / total_weight

        print(f"✅ Hybrid retriever initialized (semantic: {self.semantic_weight:.2f}, keyword: {self.keyword_weight:.2f})")

    def search(
        self,
        query: str,
        k: int = 5,
        semantic_k: int = None,
        keyword_k: int = None
    ) -> Tuple[List[str], List[float], List[dict]]:
        """
        Hybrid search combining semantic and keyword retrieval

        Args:
            query: Query text
            k: Final number of results to return
            semantic_k: Number of semantic results to fetch (default: 2*k)
            keyword_k: Number of keyword results to fetch (default: 2*k)

        Returns:
            Tuple of (texts, scores, metadata)
        """
        if semantic_k is None:
            semantic_k = k * 2
        if keyword_k is None:
            keyword_k = k * 2

        # 1. Semantic search
        query_embedding = self.embedding_model.encode_query(query)
        sem_texts, sem_scores, sem_metadata = self.vector_store.search(
            query_embedding,
            k=semantic_k
        )

        # 2. Keyword search
        kw_texts, kw_scores, kw_metadata = self.bm25_retriever.search(
            query,
            k=keyword_k
        )

        # 3. Merge results
        merged_results = self._merge_results(
            sem_texts, sem_scores, sem_metadata,
            kw_texts, kw_scores, kw_metadata
        )

        # 4. Sort by hybrid score and take top-k
        sorted_results = sorted(
            merged_results,
            key=lambda x: x['hybrid_score'],
            reverse=True
        )[:k]

        # 5. Extract final results
        final_texts = [r['text'] for r in sorted_results]
        final_scores = [r['hybrid_score'] for r in sorted_results]
        final_metadata = [r['metadata'] for r in sorted_results]

        return final_texts, final_scores, final_metadata

    def _merge_results(
        self,
        sem_texts: List[str],
        sem_scores: List[float],
        sem_metadata: List[dict],
        kw_texts: List[str],
        kw_scores: List[float],
        kw_metadata: List[dict]
    ) -> List[Dict]:
        """
        Merge and score results from both retrievers

        Args:
            sem_texts: Semantic search texts
            sem_scores: Semantic search scores
            sem_metadata: Semantic search metadata
            kw_texts: Keyword search texts
            kw_scores: Keyword search scores
            kw_metadata: Keyword search metadata

        Returns:
            List of merged results with hybrid scores
        """
        # Normalize scores to 0-1 range
        sem_scores_norm = self._normalize_scores(sem_scores)
        kw_scores_norm = self._normalize_scores(kw_scores)

        # Create result dictionary
        results_dict = {}

        # Add semantic results
        for text, score, metadata in zip(sem_texts, sem_scores_norm, sem_metadata):
            if text not in results_dict:
                results_dict[text] = {
                    'text': text,
                    'metadata': metadata,
                    'semantic_score': score,
                    'keyword_score': 0.0,
                }
            else:
                results_dict[text]['semantic_score'] = max(
                    results_dict[text]['semantic_score'],
                    score
                )

        # Add keyword results
        for text, score, metadata in zip(kw_texts, kw_scores_norm, kw_metadata):
            if text not in results_dict:
                results_dict[text] = {
                    'text': text,
                    'metadata': metadata,
                    'semantic_score': 0.0,
                    'keyword_score': score,
                }
            else:
                results_dict[text]['keyword_score'] = max(
                    results_dict[text]['keyword_score'],
                    score
                )

        # Calculate hybrid scores
        for result in results_dict.values():
            result['hybrid_score'] = (
                self.semantic_weight * result['semantic_score'] +
                self.keyword_weight * result['keyword_score']
            )

        return list(results_dict.values())

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-1 range

        Args:
            scores: List of scores

        Returns:
            Normalized scores
        """
        if not scores:
            return []

        scores_array = np.array(scores)

        min_score = scores_array.min()
        max_score = scores_array.max()

        if max_score == min_score:
            return [1.0] * len(scores)

        normalized = (scores_array - min_score) / (max_score - min_score)

        return normalized.tolist()

    def get_stats(self) -> dict:
        """Get retriever statistics"""
        return {
            'semantic_weight': self.semantic_weight,
            'keyword_weight': self.keyword_weight,
            'vector_store': self.vector_store.get_stats(),
            'bm25': self.bm25_retriever.get_stats(),
        }
