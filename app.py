"""
Pharmacy RAG Streamlit Application

A Retrieval-Augmented Generation application for answering pharmacy-related questions.
Uses LangChain, GitHub-hosted models, and ChromaDB to provide accurate answers with source citations.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from rag_utils import create_rag_system

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Pharmacy RAG Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .answer-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown('<h1 class="main-header">💊 Pharmacy RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about medications, dosages, side effects, and more!</p>', 
            unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection - using GitHub-hosted models
    model_choice = st.selectbox(
        "Select Model",
        ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
        help="Choose the GitHub-hosted model for generating responses"
    )
    
    # Initialize button
    if st.button("🚀 Initialize System", type="primary"):
        with st.spinner("Loading pharmacy knowledge base... This may take a moment."):
            try:
                # Create and initialize the RAG system
                # Uses GitHub-hosted models - credentials provided by environment
                st.session_state.rag_system = create_rag_system(
                    data_path="data/pharmacy_info.txt",
                    model_name=model_choice
                )
                st.session_state.initialized = True
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")
                st.info("💡 Note: This app uses GitHub-hosted models. Ensure you're running in a GitHub-authenticated environment (e.g., GitHub Codespaces).")
                st.session_state.initialized = False
    
    # System status
    st.divider()
    st.subheader("📊 System Status")
    if st.session_state.initialized:
        st.success("🟢 Ready")
        st.info(f"Model: {model_choice}")
    else:
        st.warning("🟡 Not initialized")
        st.info("Click 'Initialize System' to begin")
    
    # Information
    st.divider()
    st.subheader("ℹ️ About")
    st.markdown("""
    This app uses RAG (Retrieval-Augmented Generation) to answer pharmacy questions:
    
    1. **Retrieval**: Finds relevant information from the knowledge base
    2. **Augmentation**: Enhances the query with context
    3. **Generation**: Produces accurate answers using AI
    
    **Sample Questions:**
    - What are the side effects of aspirin?
    - How should insulin be stored?
    - What is the dosage for ibuprofen?
    - Drug interactions with warfarin?
    """)
    
    # Clear history button (only show if there's history)
    if st.session_state.chat_history:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

# Main content area
if st.session_state.initialized:
    # Question input
    st.subheader("💬 Ask Your Question")
    
    question = st.text_area(
        "Enter your pharmacy-related question:",
        height=100,
        placeholder="e.g., What are the side effects of aspirin?",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.question_input = ""
            st.rerun()
    
    # Process question
    if ask_button and question.strip():
        with st.spinner("Searching knowledge base and generating answer..."):
            try:
                # Query the RAG system
                result = st.session_state.rag_system.query(question)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["source_documents"]
                })
                
                # Display the answer
                st.markdown("### 📝 Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', 
                           unsafe_allow_html=True)
                
                # Display sources
                st.markdown("### 📚 Sources")
                st.info(f"Found {len(result['source_documents'])} relevant source(s)")
                
                for idx, doc in enumerate(result["source_documents"], 1):
                    with st.expander(f"Source {idx}"):
                        st.markdown(f'<div class="source-box">{doc.page_content}</div>', 
                                   unsafe_allow_html=True)
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.caption(f"Metadata: {doc.metadata}")
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
    
    elif ask_button:
        st.warning("⚠️ Please enter a question first")
    
    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        st.subheader("📜 Chat History")
        
        for idx, chat in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{len(st.session_state.chat_history) - idx + 1}: {chat['question'][:100]}..."):
                st.markdown("**Question:**")
                st.write(chat['question'])
                st.markdown("**Answer:**")
                st.info(chat['answer'])
                st.caption(f"Sources used: {len(chat['sources'])}")

else:
    # Display instructions when not initialized
    st.info("👈 Please click 'Initialize System' in the sidebar to get started.")
    
    st.markdown("### 🎯 What This App Does")
    st.markdown("""
    The Pharmacy RAG Assistant helps you get accurate answers to pharmacy-related questions by:
    
    - 🔍 **Searching** a curated database of medication information
    - 🤖 **Generating** contextual answers using AI
    - 📖 **Citing** sources for transparency and verification
    - 💡 **Providing** information on dosages, side effects, interactions, and storage
    
    ### 🚀 Getting Started
    
    1. Click "Initialize System" in the sidebar to load the knowledge base
    2. Type your question and click "Get Answer"
    3. Review the answer and check the sources used
    
    ### 🌐 GitHub-Hosted Models
    
    This application uses GitHub-hosted AI models (gpt-4o) and does not require manual API key configuration.
    When running in GitHub Codespaces or other GitHub-authenticated environments, model access is automatic.
    
    ### ⚠️ Important Disclaimer
    
    This tool is for **informational purposes only** and should not replace professional medical advice. 
    Always consult with a qualified healthcare provider or pharmacist for medical decisions.
    """)
    
    # Display sample knowledge base info
    st.markdown("### 📚 Knowledge Base Coverage")
    st.markdown("""
    The app currently includes information about:
    - Aspirin (pain relief, cardiovascular protection)
    - Ibuprofen (NSAID)
    - Insulin (diabetes management)
    - Warfarin (anticoagulant)
    - Metformin (diabetes)
    - Lisinopril (ACE inhibitor)
    - Levothyroxine (thyroid hormone)
    - Atorvastatin (statin)
    - Omeprazole (PPI)
    
    Each medication entry includes: dosage, side effects, drug interactions, and storage information.
    """)

# Footer
st.divider()
st.caption("💊 Pharmacy RAG Assistant | Built with Streamlit, LangChain, and GitHub Models | ⚠️ For informational purposes only")
