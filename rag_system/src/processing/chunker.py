"""
Text Chunker
Splits text into chunks with token overlap
"""

from typing import List, Dict
from dataclasses import dataclass
import tiktoken


@dataclass
class Chunk:
    """Text chunk with metadata"""
    text: str
    chunk_id: str
    doc_id: str
    chunk_index: int
    token_count: int
    metadata: Dict


class TextChunker:
    """
    Split text into overlapping chunks

    Uses tiktoken for accurate token counting
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"  # GPT-4, GPT-3.5-turbo encoding
    ):
        """
        Initialize chunker

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            encoding_name: Tiktoken encoding name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            print(f"⚠️  Could not load encoding {encoding_name}, falling back to word-based chunking")
            self.encoding = None

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: Dict = None
    ) -> List[Chunk]:
        """
        Split text into chunks

        Args:
            text: Text to chunk
            doc_id: Document ID
            metadata: Document metadata

        Returns:
            List of Chunk objects
        """
        if metadata is None:
            metadata = {}

        if self.encoding is not None:
            return self._chunk_with_tiktoken(text, doc_id, metadata)
        else:
            return self._chunk_with_words(text, doc_id, metadata)

    def _chunk_with_tiktoken(self, text: str, doc_id: str, metadata: Dict) -> List[Chunk]:
        """Chunk using tiktoken for accurate token counting"""
        # Encode text to tokens
        tokens = self.encoding.encode(text)

        chunks = []
        start_idx = 0
        chunk_index = 0

        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = start_idx + self.chunk_size
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)

            # Create chunk
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                chunk_index=chunk_index,
                token_count=len(chunk_tokens),
                metadata={**metadata, 'start_token': start_idx, 'end_token': end_idx}
            )

            chunks.append(chunk)

            # Move to next chunk with overlap
            start_idx += self.chunk_size - self.chunk_overlap
            chunk_index += 1

        return chunks

    def _chunk_with_words(self, text: str, doc_id: str, metadata: Dict) -> List[Chunk]:
        """Fallback: chunk using word count (approximation)"""
        words = text.split()

        # Approximate: 1 token ≈ 0.75 words
        words_per_chunk = int(self.chunk_size * 0.75)
        words_overlap = int(self.chunk_overlap * 0.75)

        chunks = []
        start_idx = 0
        chunk_index = 0

        while start_idx < len(words):
            # Get chunk words
            end_idx = start_idx + words_per_chunk
            chunk_words = words[start_idx:end_idx]

            # Join back to text
            chunk_text = ' '.join(chunk_words)

            # Create chunk
            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                chunk_index=chunk_index,
                token_count=len(chunk_words),  # Approximate
                metadata={**metadata, 'start_word': start_idx, 'end_word': end_idx}
            )

            chunks.append(chunk)

            # Move to next chunk with overlap
            start_idx += words_per_chunk - words_overlap
            chunk_index += 1

        return chunks

    def chunk_documents(self, documents: List) -> List[Chunk]:
        """
        Chunk multiple documents

        Args:
            documents: List of Document objects

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_text(
                text=doc.content,
                doc_id=doc.doc_id,
                metadata=doc.metadata
            )
            all_chunks.extend(chunks)

        return all_chunks

    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict:
        """
        Get statistics about chunks

        Args:
            chunks: List of chunks

        Returns:
            Dict with statistics
        """
        if not chunks:
            return {}

        token_counts = [c.token_count for c in chunks]

        return {
            'total_chunks': len(chunks),
            'avg_tokens_per_chunk': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'total_tokens': sum(token_counts),
        }
