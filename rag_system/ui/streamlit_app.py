"""
Streamlit WebUI for RAG System
Beautiful and intuitive interface for querying documents
"""

import streamlit as st
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.server import RAGSystem


# Page config
st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.stButton>button {
    width: 100%;
    background-color: #1f77b4;
    color: white;
    border-radius: 5px;
    height: 3rem;
    font-size: 1.2rem;
}
.context-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 5px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.initialized = False
    st.session_state.query_history = []


def initialize_system(model_name: str, vector_store_path: str = None, load_in_4bit: bool = True):
    """Initialize RAG system"""
    with st.spinner("🚀 Initializing RAG system... This may take a few minutes..."):
        try:
            rag_system = RAGSystem()
            rag_system.initialize(
                vector_store_path=vector_store_path if vector_store_path else None,
                model_name=model_name,
                load_in_4bit=load_in_4bit
            )
            st.session_state.rag_system = rag_system
            st.session_state.initialized = True
            st.success("✅ RAG system initialized successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            return False


def index_documents(documents_path: str):
    """Index documents"""
    if not st.session_state.initialized:
        st.error("⚠️ Please initialize the system first!")
        return

    with st.spinner("📚 Indexing documents..."):
        try:
            st.session_state.rag_system.index_documents(documents_path)
            st.success(f"✅ Documents indexed successfully!")

            # Show stats
            stats = st.session_state.rag_system.vector_store.get_stats()
            st.info(f"📊 Total vectors in store: {stats['total_vectors']:,}")
        except Exception as e:
            st.error(f"❌ Indexing failed: {str(e)}")


def query_system(query: str, k: int, temperature: float, max_tokens: int):
    """Query RAG system"""
    if not st.session_state.initialized:
        st.error("⚠️ Please initialize the system first!")
        return None

    try:
        start_time = time.time()
        result = st.session_state.rag_system.query(
            query=query,
            k=k,
            temperature=temperature,
            max_tokens=max_tokens
        )
        result['processing_time'] = time.time() - start_time

        # Add to history
        st.session_state.query_history.append({
            'query': query,
            'answer': result['answer'],
            'time': result['processing_time']
        })

        return result
    except Exception as e:
        st.error(f"❌ Query failed: {str(e)}")
        return None


# Main UI
st.markdown('<h1 class="main-header">🔍 RAG System</h1>', unsafe_allow_html=True)
st.markdown("### Retrieval Augmented Generation for Document Q&A")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # System initialization
    st.subheader("1. Initialize System")

    model_name = st.selectbox(
        "LLM Model",
        [
            "mistralai/Mistral-Nemo-Instruct-2407",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x22B-Instruct-v0.1"
        ],
        index=0
    )

    load_in_4bit = st.checkbox("Use 4-bit quantization", value=True,
                                help="Reduces memory usage significantly")

    vector_store_path = st.text_input("Vector Store Path (optional)",
                                       value="",
                                       placeholder="data/vector_store")

    if st.button("🚀 Initialize System"):
        initialize_system(
            model_name=model_name,
            vector_store_path=vector_store_path if vector_store_path else None,
            load_in_4bit=load_in_4bit
        )

    st.markdown("---")

    # Document indexing
    st.subheader("2. Index Documents")

    documents_path = st.text_input("Documents Directory",
                                    value="",
                                    placeholder="rag_system/data/documents")

    if st.button("📚 Index Documents"):
        if documents_path:
            index_documents(documents_path)
        else:
            st.error("Please enter a documents path")

    st.markdown("---")

    # Query parameters
    st.subheader("3. Query Parameters")

    k = st.slider("Number of context chunks", min_value=1, max_value=10, value=5)
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max tokens", min_value=50, max_value=2048, value=512, step=50)

    st.markdown("---")

    # System status
    st.subheader("📊 System Status")

    if st.session_state.initialized:
        st.success("✅ System initialized")

        stats = st.session_state.rag_system.vector_store.get_stats()
        st.metric("Documents indexed", f"{stats['total_vectors']:,}")
        st.metric("Queries processed", len(st.session_state.query_history))
    else:
        st.warning("⚠️ System not initialized")


# Main area - Query interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask a Question")

    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="What is the main topic of the documents?"
    )

    if st.button("🔍 Search", key="search_button"):
        if query:
            with st.spinner("🤔 Thinking..."):
                result = query_system(query, k, temperature, max_tokens)

                if result:
                    # Display answer
                    st.markdown("### 📝 Answer")
                    st.markdown(f"**{result['answer']}**")

                    st.markdown(f"⏱️ *Processing time: {result['processing_time']:.2f}s*")

                    # Display context
                    st.markdown("### 📚 Retrieved Context")
                    for i, (context, score) in enumerate(zip(result['context'], result['scores']), 1):
                        with st.expander(f"Context {i} (Score: {score:.3f})"):
                            st.markdown(f'<div class="context-box">{context}</div>', unsafe_allow_html=True)
        else:
            st.error("Please enter a question")

with col2:
    st.subheader("📜 Query History")

    if st.session_state.query_history:
        for i, item in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.expander(f"Query {len(st.session_state.query_history) - i + 1}"):
                st.markdown(f"**Q:** {item['query']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"⏱️ {item['time']:.2f}s")

        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.experimental_rerun()
    else:
        st.info("No queries yet. Ask a question to get started!")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>RAG System v1.0 | Built with Streamlit, FAISS, and Transformers</p>
</div>
""", unsafe_allow_html=True)
