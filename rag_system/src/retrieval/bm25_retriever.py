"""
BM25 Retriever
Keyword-based retrieval using BM25 algorithm
"""

from typing import List, Tuple
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path


class BM25Retriever:
    """
    BM25-based keyword retriever

    Complements semantic search with keyword matching
    """

    def __init__(self, tokenizer=None):
        """
        Initialize BM25 retriever

        Args:
            tokenizer: Optional custom tokenizer function
        """
        self.tokenizer = tokenizer or self._default_tokenizer
        self.bm25 = None
        self.texts: List[str] = []
        self.metadata: List[dict] = []
        self.doc_ids: List[str] = []

    def _default_tokenizer(self, text: str) -> List[str]:
        """
        Default tokenizer: lowercase and split

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return text.lower().split()

    def add_texts(
        self,
        texts: List[str],
        metadata: List[dict] = None,
        doc_ids: List[str] = None
    ):
        """
        Add texts to BM25 index

        Args:
            texts: List of text chunks
            metadata: Optional metadata for each text
            doc_ids: Optional document IDs
        """
        # Tokenize all texts
        tokenized_corpus = [self.tokenizer(text) for text in texts]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Store texts and metadata
        self.texts = texts

        if metadata:
            self.metadata = metadata
        else:
            self.metadata = [{}] * len(texts)

        if doc_ids:
            self.doc_ids = doc_ids
        else:
            self.doc_ids = [''] * len(texts)

        print(f"✅ BM25 index built ({len(texts)} documents)")

    def search(
        self,
        query: str,
        k: int = 5
    ) -> Tuple[List[str], List[float], List[dict]]:
        """
        Search using BM25

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            Tuple of (texts, scores, metadata)
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not initialized. Call add_texts() first.")

        # Tokenize query
        query_tokens = self.tokenizer(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_k_indices = scores.argsort()[-k:][::-1]

        # Extract results
        results_texts = [self.texts[i] for i in top_k_indices]
        results_scores = [float(scores[i]) for i in top_k_indices]
        results_metadata = [self.metadata[i] for i in top_k_indices]

        return results_texts, results_scores, results_metadata

    def save(self, save_path: str):
        """
        Save BM25 index to disk

        Args:
            save_path: Directory to save to
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        data = {
            'bm25': self.bm25,
            'texts': self.texts,
            'metadata': self.metadata,
            'doc_ids': self.doc_ids,
        }

        with open(save_dir / "bm25.pkl", 'wb') as f:
            pickle.dump(data, f)

        print(f"✅ BM25 index saved to {save_path}")

    @classmethod
    def load(cls, load_path: str) -> 'BM25Retriever':
        """
        Load BM25 index from disk

        Args:
            load_path: Directory to load from

        Returns:
            BM25Retriever instance
        """
        load_dir = Path(load_path)

        if not load_dir.exists():
            raise FileNotFoundError(f"BM25 index not found: {load_path}")

        with open(load_dir / "bm25.pkl", 'rb') as f:
            data = pickle.load(f)

        retriever = cls()
        retriever.bm25 = data['bm25']
        retriever.texts = data['texts']
        retriever.metadata = data['metadata']
        retriever.doc_ids = data['doc_ids']

        print(f"✅ BM25 index loaded from {load_path} ({len(retriever.texts)} documents)")

        return retriever

    def get_stats(self) -> dict:
        """Get BM25 retriever statistics"""
        return {
            'total_documents': len(self.texts),
            'avg_doc_length': self.bm25.avgdl if self.bm25 else 0,
        }
