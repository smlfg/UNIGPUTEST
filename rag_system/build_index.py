#!/usr/bin/env python3
"""
Build RAG Index
Process documents and build vector store index
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.processing import DocumentProcessor
from src.retrieval import EmbeddingModel, VectorStore, BM25Retriever


def build_index(
    documents_path: str,
    output_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2"
):
    """
    Build vector store index from documents

    Args:
        documents_path: Path to documents directory
        output_path: Path to save vector store
        chunk_size: Chunk size in tokens
        chunk_overlap: Overlap between chunks
        embedding_model: Embedding model name
    """
    print("="*80)
    print("BUILDING RAG INDEX")
    print("="*80)

    # 1. Process documents
    print(f"\n1️⃣  Processing documents from: {documents_path}")
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = processor.process_directory(documents_path, recursive=True)

    if not chunks:
        print("❌ No documents found!")
        return

    # Extract data
    texts = [chunk.text for chunk in chunks]
    metadata = [chunk.metadata for chunk in chunks]
    doc_ids = [chunk.doc_id for chunk in chunks]

    # 2. Initialize embedding model
    print(f"\n2️⃣  Loading embedding model: {embedding_model}")
    embed_model = EmbeddingModel(model_name=embedding_model)

    # 3. Generate embeddings
    print(f"\n3️⃣  Generating embeddings for {len(texts)} chunks...")
    embeddings = embed_model.encode_documents(texts, show_progress=True)

    # 4. Build FAISS index
    print(f"\n4️⃣  Building FAISS vector store...")
    vector_store = VectorStore(
        dimension=embed_model.dimension,
        index_type="flat",
        metric="cosine"
    )
    vector_store.add_texts(texts, embeddings, metadata, doc_ids)

    # 5. Build BM25 index
    print(f"\n5️⃣  Building BM25 index...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.add_texts(texts, metadata, doc_ids)

    # 6. Save indices
    print(f"\n6️⃣  Saving indices to: {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    vector_store.save(output_path)
    bm25_retriever.save(output_path)

    # Summary
    print("\n" + "="*80)
    print("✅ INDEX BUILT SUCCESSFULLY")
    print("="*80)
    print(f"Total chunks indexed:  {len(texts):,}")
    print(f"Total tokens:          {sum(chunk.token_count for chunk in chunks):,}")
    print(f"Embedding dimension:   {embed_model.dimension}")
    print(f"Output path:           {output_path}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from documents")
    parser.add_argument(
        "--documents",
        type=str,
        required=True,
        help="Path to documents directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/vector_store",
        help="Path to save vector store (default: data/vector_store)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens (default: 512)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)"
    )

    args = parser.parse_args()

    build_index(
        documents_path=args.documents,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model
    )


if __name__ == "__main__":
    main()
