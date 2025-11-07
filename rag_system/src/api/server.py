"""
FastAPI Server for RAG System
Production-ready API with async endpoints and rate limiting
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.processing import DocumentProcessor
from src.retrieval import EmbeddingModel, VectorStore, BM25Retriever, HybridRetriever
from src.generation import LLMGenerator, RAGPromptBuilder


# Pydantic models for API
class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., description="User query", min_length=1, max_length=1000)
    k: int = Field(5, description="Number of results to retrieve", ge=1, le=20)
    temperature: float = Field(0.7, description="Generation temperature", ge=0.0, le=2.0)
    max_tokens: int = Field(512, description="Maximum tokens to generate", ge=50, le=2048)


class QueryResponse(BaseModel):
    """Query response model"""
    query: str
    answer: str
    context: List[str]
    scores: List[float]
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    vector_store_size: int


class StatsResponse(BaseModel):
    """Statistics response"""
    vector_store: Dict
    bm25: Dict
    model_info: Dict


# Rate limiting
class RateLimiter:
    """Simple rate limiter"""

    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests = {}

    async def __call__(self, request: Request):
        client_ip = request.client.host
        current_time = time.time()

        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60
            ]
        else:
            self.requests[client_ip] = []

        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.requests_per_minute} requests per minute."
            )

        # Add current request
        self.requests[client_ip].append(current_time)


# RAG System
class RAGSystem:
    """Complete RAG system"""

    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.llm = None
        self.prompt_builder = None
        self.initialized = False

    def initialize(
        self,
        vector_store_path: str = None,
        model_name: str = "mistralai/Mistral-Nemo-Instruct-2407",
        load_in_4bit: bool = True
    ):
        """
        Initialize RAG system

        Args:
            vector_store_path: Path to saved vector store (optional)
            model_name: LLM model name
            load_in_4bit: Use 4-bit quantization
        """
        print("Initializing RAG system...")

        # 1. Embedding model
        self.embedding_model = EmbeddingModel()

        # 2. Vector store
        if vector_store_path and Path(vector_store_path).exists():
            print(f"Loading vector store from {vector_store_path}")
            self.vector_store = VectorStore.load(vector_store_path)
            self.bm25_retriever = BM25Retriever.load(vector_store_path)
        else:
            print("Creating new vector store")
            self.vector_store = VectorStore(dimension=self.embedding_model.dimension)
            self.bm25_retriever = BM25Retriever()

        # 3. Hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever
        )

        # 4. LLM
        print(f"Loading LLM: {model_name}")
        self.llm = LLMGenerator(
            model_name=model_name,
            load_in_4bit=load_in_4bit
        )

        # 5. Prompt builder
        self.prompt_builder = RAGPromptBuilder(template_type="mistral")

        self.initialized = True
        print("✅ RAG system initialized successfully")

    def query(
        self,
        query: str,
        k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict:
        """
        Process RAG query

        Args:
            query: User query
            k: Number of context chunks to retrieve
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with answer, context, and metadata
        """
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")

        start_time = time.time()

        # 1. Retrieve context
        context_texts, scores, metadata = self.hybrid_retriever.search(
            query=query,
            k=k
        )

        # 2. Generate response
        answer = self.llm.generate_rag_response(
            query=query,
            context=context_texts,
            temperature=temperature,
            max_new_tokens=max_tokens
        )

        processing_time = time.time() - start_time

        return {
            'query': query,
            'answer': answer,
            'context': context_texts,
            'scores': scores,
            'processing_time': processing_time
        }

    def index_documents(self, documents_path: str):
        """
        Index documents from directory

        Args:
            documents_path: Path to documents directory
        """
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")

        # Process documents
        processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
        chunks = processor.process_directory(documents_path)

        if not chunks:
            raise ValueError("No documents found to index")

        # Extract texts and metadata
        texts = [chunk.text for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        doc_ids = [chunk.doc_id for chunk in chunks]

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode_documents(texts, show_progress=True)

        # Add to vector store
        self.vector_store.add_texts(texts, embeddings, metadata, doc_ids)

        # Add to BM25
        self.bm25_retriever.add_texts(texts, metadata, doc_ids)

        print(f"✅ Indexed {len(texts)} chunks")

    def save(self, save_path: str):
        """Save RAG system state"""
        self.vector_store.save(save_path)
        self.bm25_retriever.save(save_path)
        print(f"✅ RAG system saved to {save_path}")


# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Production-ready Retrieval Augmented Generation system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter
rate_limiter = RateLimiter(requests_per_minute=10)

# Global RAG system instance
rag_system = RAGSystem()


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    # You can customize initialization here
    # rag_system.initialize(vector_store_path="data/vector_store")
    pass


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "models_loaded": rag_system.initialized,
        "vector_store_size": rag_system.vector_store.index.ntotal if rag_system.initialized else 0
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if rag_system.initialized else "not_initialized",
        "models_loaded": rag_system.initialized,
        "vector_store_size": rag_system.vector_store.index.ntotal if rag_system.initialized else 0
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    if not rag_system.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    return {
        "vector_store": rag_system.vector_store.get_stats(),
        "bm25": rag_system.bm25_retriever.get_stats(),
        "model_info": rag_system.llm.get_model_info()
    }


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limiter)])
async def query(request: QueryRequest):
    """
    Query the RAG system

    Args:
        request: Query request with query text and parameters

    Returns:
        Query response with answer and context
    """
    if not rag_system.initialized:
        raise HTTPException(status_code=503, detail="System not initialized. Please initialize first.")

    try:
        result = rag_system.query(
            query=request.query,
            k=request.k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return QueryResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/initialize")
async def initialize_system(
    model_name: str = "mistralai/Mistral-Nemo-Instruct-2407",
    vector_store_path: Optional[str] = None,
    load_in_4bit: bool = True
):
    """
    Initialize RAG system

    Args:
        model_name: LLM model name
        vector_store_path: Path to saved vector store
        load_in_4bit: Use 4-bit quantization

    Returns:
        Initialization status
    """
    try:
        rag_system.initialize(
            vector_store_path=vector_store_path,
            model_name=model_name,
            load_in_4bit=load_in_4bit
        )

        return {"status": "initialized", "model": model_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.post("/index")
async def index_documents(documents_path: str):
    """
    Index documents from directory

    Args:
        documents_path: Path to documents directory

    Returns:
        Indexing status
    """
    if not rag_system.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        rag_system.index_documents(documents_path)
        return {"status": "indexed", "path": documents_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
