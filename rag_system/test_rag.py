#!/usr/bin/env python3
"""
Test RAG System
Interactive testing of the RAG pipeline
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.api.server import RAGSystem


def test_rag_system(
    vector_store_path: str = None,
    model_name: str = "mistralai/Mistral-Nemo-Instruct-2407",
    load_in_4bit: bool = True
):
    """
    Test RAG system interactively

    Args:
        vector_store_path: Path to vector store
        model_name: LLM model name
        load_in_4bit: Use 4-bit quantization
    """
    print("\n" + "="*80)
    print("RAG SYSTEM TEST")
    print("="*80 + "\n")

    # Initialize system
    print("Initializing RAG system...")
    rag_system = RAGSystem()

    try:
        rag_system.initialize(
            vector_store_path=vector_store_path,
            model_name=model_name,
            load_in_4bit=load_in_4bit
        )
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Interactive query loop
    print("\n" + "="*80)
    print("READY FOR QUERIES")
    print("="*80)
    print("Type your questions below. Type 'quit' to exit.\n")

    while True:
        try:
            # Get query
            query = input("\n❓ Your question: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not query:
                continue

            # Process query
            print("\n🤔 Thinking...")

            result = rag_system.query(
                query=query,
                k=5,
                temperature=0.7,
                max_tokens=512
            )

            # Display results
            print("\n" + "─"*80)
            print("📝 ANSWER")
            print("─"*80)
            print(result['answer'])

            print("\n" + "─"*80)
            print("📚 RETRIEVED CONTEXT")
            print("─"*80)
            for i, (context, score) in enumerate(zip(result['context'], result['scores']), 1):
                print(f"\n[{i}] Score: {score:.3f}")
                print(f"{context[:200]}...")

            print(f"\n⏱️  Processing time: {result['processing_time']:.2f}s")
            print("─"*80)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test RAG system")
    parser.add_argument(
        "--vector-store",
        type=str,
        default=None,
        help="Path to vector store (optional)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-Nemo-Instruct-2407",
        help="LLM model name"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )

    args = parser.parse_args()

    test_rag_system(
        vector_store_path=args.vector_store,
        model_name=args.model,
        load_in_4bit=not args.no_4bit
    )


if __name__ == "__main__":
    main()
