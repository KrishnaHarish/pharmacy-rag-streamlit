"""
Pharmacy RAG Streamlit Application - Self-Contained Local Version

A fully local Retrieval-Augmented Generation application for pharmacy questions.
Uses llama-cpp-python with GGUF models, sentence-transformers embeddings, and FAISS.
No external API keys required - runs entirely on Streamlit Community Cloud.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from pathlib import Path
from typing import List, Tuple
import textwrap

# Set page config first
st.set_page_config(
    page_title="Pharmacy RAG Assistant (Local)",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)
MODEL_PATH = os.getenv("MODEL_PATH", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DRUGS_CSV = "drugs.csv"

# CSS Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def download_model(model_url: str, model_path: str) -> str:
    """
    Download GGUF model if not present. Cached to avoid re-downloading.
    
    Args:
        model_url: URL to download the GGUF model from
        model_path: Local path to save the model
        
    Returns:
        Path to the downloaded model
    """
    model_file = Path(model_path)
    
    # Create directory if it doesn't exist
    model_file.parent.mkdir(parents=True, exist_ok=True)
    
    if model_file.exists():
        st.info(f"✓ Model found at {model_path}")
        return str(model_file)
    
    st.info(f"Downloading model from {model_url}...")
    st.info("This may take a few minutes on first run...")
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        try:
            total_size = int(response.headers.get('content-length', 0))
        except (ValueError, TypeError):
            st.warning("Could not determine file size, progress bar may be inaccurate")
            total_size = 0
        
        progress_bar = st.progress(0)
        downloaded = 0
        
        with open(model_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(min(downloaded / total_size, 1.0))
        
        progress_bar.empty()
        st.success(f"✓ Model downloaded to {model_path}")
        return str(model_file)
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error downloading model: {e}")
        st.info("Please check your internet connection and MODEL_URL")
        raise
        
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.info("Please check MODEL_URL or pre-bundle the model in the models/ directory")
        raise


@st.cache_resource
def load_llm(model_path: str, n_ctx: int = 2048, n_threads: int = 4, verbose: bool = False):
    """
    Load the GGUF model with llama-cpp-python. Cached for performance.
    
    Args:
        model_path: Path to the GGUF model file
        n_ctx: Context window size
        n_threads: Number of CPU threads to use
        verbose: Enable verbose logging for debugging
        
    Returns:
        Llama model instance
    """
    try:
        from llama_cpp import Llama
        
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,  # CPU only for Streamlit Cloud
            verbose=verbose
        )
        
        st.success("✓ LLM loaded successfully")
        return llm
        
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        st.info("Install llama-cpp-python: pip install llama-cpp-python")
        raise


@st.cache_resource
def load_embeddings():
    """
    Load sentence-transformers embedding model. Cached for performance.
    
    Returns:
        SentenceTransformer model instance
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        st.success("✓ Embeddings model loaded successfully")
        return embedder
        
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        raise


@st.cache_resource
def build_faiss_index(_embedder, drugs_df: pd.DataFrame):
    """
    Build FAISS index from drugs.csv. Cached for performance.
    
    Args:
        _embedder: SentenceTransformer model (underscore prefix to avoid hashing)
        drugs_df: DataFrame with drug and description columns
        
    Returns:
        Tuple of (faiss_index, drug_texts, drug_names)
    """
    try:
        import faiss
        
        # Prepare texts for embedding
        drug_texts = []
        drug_names = []
        
        for _, row in drugs_df.iterrows():
            text = f"{row['drug']}: {row['description']}"
            drug_texts.append(text)
            drug_names.append(row['drug'])
        
        # Generate embeddings
        st.info(f"Generating embeddings for {len(drug_texts)} drugs...")
        embeddings = _embedder.encode(drug_texts, show_progress_bar=False)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product = cosine for normalized vectors
        index.add(embeddings.astype('float32'))
        
        st.success(f"✓ FAISS index built with {len(drug_texts)} entries")
        return index, drug_texts, drug_names
        
    except Exception as e:
        st.error(f"Error building FAISS index: {e}")
        raise


def retrieve_context(query: str, embedder, faiss_index, drug_texts: List[str], 
                     drug_names: List[str], top_k: int = 3) -> List[Tuple[str, str, float]]:
    """
    Retrieve top-k relevant drug descriptions using FAISS.
    
    Args:
        query: User question
        embedder: SentenceTransformer model
        faiss_index: FAISS index
        drug_texts: List of drug description texts
        drug_names: List of drug names
        top_k: Number of results to return
        
    Returns:
        List of tuples (drug_name, description, score)
    """
    # Encode query
    query_embedding = embedder.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # Search FAISS
    scores, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if 0 <= idx < len(drug_texts):
            results.append((drug_names[idx], drug_texts[idx], float(score)))
        else:
            # Log warning if FAISS returns invalid index (shouldn't happen with properly built index)
            import warnings
            warnings.warn(f"FAISS returned invalid index {idx}, skipping (index size: {len(drug_texts)})")
    
    return results


def build_prompt(query: str, context_items: List[Tuple[str, str, float]]) -> str:
    """
    Build prompt with context snippets for the LLM.
    
    Args:
        query: User question
        context_items: List of (drug_name, description, score) tuples
        
    Returns:
        Formatted prompt string
    """
    context_str = ""
    for i, (drug_name, text, score) in enumerate(context_items, 1):
        # Wrap long text
        wrapped = textwrap.fill(text, width=100)
        context_str += f"[S{i}] {wrapped}\n\n"
    
    prompt = f"""<|system|>
You are a knowledgeable pharmacy assistant. Answer questions based on the provided context.
Use [S#] citations to reference sources. Be concise and accurate.
</|system|>

<|user|>
Context:
{context_str}

Question: {query}

Provide a concise answer with [S#] citations.
</|user|>

<|assistant|>
"""
    return prompt


def generate_answer(llm, prompt: str, max_tokens: int = 400, temperature: float = 0.7) -> str:
    """
    Generate answer using the LLM.
    
    Args:
        llm: Llama model instance
        prompt: Formatted prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated answer text
    """
    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop=["</|assistant|>", "<|user|>"],
            echo=False
        )
        
        answer = output['choices'][0]['text'].strip()
        return answer
        
    except Exception as e:
        return f"Error generating answer: {e}"


# Main App
def main():
    st.markdown('<h1 class="main-header">💊 Pharmacy RAG Assistant (Local)</h1>', 
                unsafe_allow_html=True)
    st.markdown("*Powered by local GGUF model, sentence-transformers, and FAISS*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Model Settings")
        st.text_input("Model URL", value=MODEL_URL, disabled=True, 
                     help="Set MODEL_URL environment variable to change")
        st.text_input("Model Path", value=MODEL_PATH, disabled=True,
                     help="Set MODEL_PATH environment variable to change")
        
        n_ctx = st.slider("Context Window", 512, 4096, 2048, 256,
                         help="Model context size")
        n_threads = st.slider("CPU Threads", 1, 8, 4, 1,
                             help="Number of CPU threads for inference")
        
        st.subheader("Generation Settings")
        top_k = st.slider("Top-K Results", 1, 10, 3, 1,
                         help="Number of drug descriptions to retrieve")
        max_tokens = st.slider("Max Tokens", 100, 800, 400, 50,
                              help="Maximum tokens in response")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
                               help="Sampling temperature (lower = more focused)")
        
        st.divider()
        st.subheader("ℹ️ About")
        st.markdown("""
        **Self-Contained RAG System:**
        - ✅ No API keys required
        - ✅ Runs entirely locally
        - ✅ Uses TinyLlama GGUF model
        - ✅ FAISS vector search
        - ✅ Sentence transformers embeddings
        
        **First Run:**
        Model downloads automatically (~640MB).
        Subsequent runs load from cache.
        """)
    
    # Initialize system
    try:
        # Step 1: Download/Load model
        with st.spinner("Loading model..."):
            model_file = download_model(MODEL_URL, MODEL_PATH)
        
        # Step 2: Load LLM
        with st.spinner("Initializing LLM..."):
            verbose_mode = os.getenv("LLM_VERBOSE", "false").lower() == "true"
            llm = load_llm(model_file, n_ctx=n_ctx, n_threads=n_threads, verbose=verbose_mode)
        
        # Step 3: Load embeddings
        with st.spinner("Loading embeddings..."):
            embedder = load_embeddings()
        
        # Step 4: Load drugs data
        if not os.path.exists(DRUGS_CSV):
            st.error(f"Error: {DRUGS_CSV} not found!")
            st.stop()
        
        drugs_df = pd.read_csv(DRUGS_CSV)
        st.success(f"✓ Loaded {len(drugs_df)} drugs from {DRUGS_CSV}")
        
        # Step 5: Build FAISS index
        with st.spinner("Building FAISS index..."):
            faiss_index, drug_texts, drug_names = build_faiss_index(embedder, drugs_df)
        
        st.success("✓ System initialized and ready!")
        
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()
    
    # Query interface
    st.divider()
    st.subheader("💬 Ask Your Question")
    
    question = st.text_area(
        "Enter your pharmacy-related question:",
        height=100,
        placeholder="e.g., What are the side effects of aspirin?",
        help="Ask about dosages, side effects, interactions, or storage"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        generate_btn = st.button("🔍 Generate Answer", type="primary", use_container_width=True)
    
    if generate_btn and question.strip():
        with st.spinner("Retrieving relevant information..."):
            # Retrieve context
            context_items = retrieve_context(
                question, embedder, faiss_index, drug_texts, drug_names, top_k
            )
        
        st.info(f"Retrieved {len(context_items)} relevant sources")
        
        # Display sources
        with st.expander("📚 Retrieved Sources", expanded=False):
            for i, (drug_name, text, score) in enumerate(context_items, 1):
                st.markdown(f"**[S{i}] {drug_name}** (similarity: {score:.3f})")
                st.markdown(f'<div class="source-box">{text}</div>', 
                           unsafe_allow_html=True)
        
        with st.spinner("Generating answer..."):
            # Build prompt
            prompt = build_prompt(question, context_items)
            
            # Generate answer
            answer = generate_answer(llm, prompt, max_tokens, temperature)
        
        # Display answer
        st.markdown("### 📝 Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', 
                   unsafe_allow_html=True)
        
    elif generate_btn:
        st.warning("⚠️ Please enter a question first")
    
    # Footer
    st.divider()
    st.caption("💊 Pharmacy RAG Assistant (Local) | Self-contained deployment for Streamlit Community Cloud")
    st.caption("⚠️ For informational purposes only - consult healthcare professionals for medical advice")


if __name__ == "__main__":
    main()
