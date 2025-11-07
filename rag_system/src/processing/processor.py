"""
Document Processor
Combines loading, chunking, and deduplication
"""

from typing import List, Set
import hashlib
from .document_loader import DocumentLoader, Document
from .chunker import TextChunker, Chunk


class DocumentProcessor:
    """
    Complete document processing pipeline

    Workflow:
    1. Load documents from files/directory
    2. Deduplicate based on content hash
    3. Chunk into smaller pieces with overlap
    4. Return processed chunks ready for embedding
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        enable_deduplication: bool = True
    ):
        """
        Initialize processor

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            enable_deduplication: Whether to deduplicate documents
        """
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.enable_deduplication = enable_deduplication

        self.seen_hashes: Set[str] = set()
        self.stats = {
            'documents_loaded': 0,
            'documents_deduplicated': 0,
            'chunks_created': 0,
        }

    def process_file(self, file_path: str) -> List[Chunk]:
        """
        Process a single file

        Args:
            file_path: Path to file

        Returns:
            List of chunks
        """
        # Load document
        doc = self.loader.load_file(file_path)
        if not doc:
            return []

        self.stats['documents_loaded'] += 1

        # Deduplicate
        if self.enable_deduplication:
            content_hash = self._hash_content(doc.content)
            if content_hash in self.seen_hashes:
                self.stats['documents_deduplicated'] += 1
                print(f"⚠️  Skipping duplicate: {file_path}")
                return []
            self.seen_hashes.add(content_hash)

        # Chunk
        chunks = self.chunker.chunk_text(
            text=doc.content,
            doc_id=doc.doc_id,
            metadata=doc.metadata
        )

        self.stats['chunks_created'] += len(chunks)

        return chunks

    def process_directory(
        self,
        directory_path: str,
        recursive: bool = True
    ) -> List[Chunk]:
        """
        Process all documents in directory

        Args:
            directory_path: Path to directory
            recursive: Recursively search subdirectories

        Returns:
            List of all chunks from all documents
        """
        # Load all documents
        documents = self.loader.load_directory(directory_path, recursive=recursive)

        if not documents:
            print(f"⚠️  No documents found in {directory_path}")
            return []

        self.stats['documents_loaded'] = len(documents)

        # Deduplicate
        if self.enable_deduplication:
            documents = self._deduplicate_documents(documents)

        # Chunk all documents
        all_chunks = self.chunker.chunk_documents(documents)
        self.stats['chunks_created'] = len(all_chunks)

        # Print statistics
        chunk_stats = self.chunker.get_chunk_stats(all_chunks)
        self.print_stats(chunk_stats)

        return all_chunks

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Remove duplicate documents based on content hash

        Args:
            documents: List of documents

        Returns:
            Deduplicated list
        """
        unique_docs = []

        for doc in documents:
            content_hash = self._hash_content(doc.content)

            if content_hash not in self.seen_hashes:
                self.seen_hashes.add(content_hash)
                unique_docs.append(doc)
            else:
                self.stats['documents_deduplicated'] += 1

        if self.stats['documents_deduplicated'] > 0:
            print(f"🗑️  Removed {self.stats['documents_deduplicated']} duplicate documents")

        return unique_docs

    def _hash_content(self, content: str) -> str:
        """Generate hash of content for deduplication"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def print_stats(self, chunk_stats: dict = None):
        """Print processing statistics"""
        print("\n" + "="*70)
        print("📊 DOCUMENT PROCESSING STATISTICS")
        print("="*70)
        print(f"Documents loaded:       {self.stats['documents_loaded']}")
        print(f"Duplicates removed:     {self.stats['documents_deduplicated']}")
        print(f"Unique documents:       {self.stats['documents_loaded'] - self.stats['documents_deduplicated']}")
        print(f"Total chunks created:   {self.stats['chunks_created']}")

        if chunk_stats:
            print(f"\nChunk Statistics:")
            print(f"  Avg tokens/chunk:     {chunk_stats.get('avg_tokens_per_chunk', 0):.1f}")
            print(f"  Min tokens:           {chunk_stats.get('min_tokens', 0)}")
            print(f"  Max tokens:           {chunk_stats.get('max_tokens', 0)}")
            print(f"  Total tokens:         {chunk_stats.get('total_tokens', 0):,}")

        print("="*70 + "\n")

    def reset_stats(self):
        """Reset statistics and seen hashes"""
        self.stats = {
            'documents_loaded': 0,
            'documents_deduplicated': 0,
            'chunks_created': 0,
        }
        self.seen_hashes.clear()
