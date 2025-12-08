"""
RAG (Retrieval-Augmented Generation) utilities for the Pharmacy application.
Handles document loading, embedding, vector store creation, and query processing.
Uses GitHub-hosted models for AI capabilities.
"""

import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class PharmacyRAG:
    """
    A RAG system for pharmacy-related question answering.
    Uses GitHub-hosted models for embeddings and generation.
    """
    
    def __init__(self, data_path: str = "data/pharmacy_info.txt"):
        """
        Initialize the Pharmacy RAG system.
        
        Args:
            data_path: Path to the pharmacy information text file
        """
        self.data_path = data_path
        self.vectorstore = None
        self.qa_chain = None
        
    def load_and_process_documents(self) -> List[Any]:
        """
        Load documents from the data file and split into chunks.
        
        Returns:
            List of document chunks
        """
        # Load the document
        loader = TextLoader(self.data_path, encoding='utf-8')
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    
    def create_vectorstore(self, documents: List[Any]) -> None:
        """
        Create a vector store from document chunks.
        Uses GitHub-hosted embeddings model.
        
        Args:
            documents: List of document chunks to embed
        """
        # Create embeddings using GitHub-hosted models
        # Credentials are provided by the environment (GitHub Codespaces, etc.)
        embeddings = OpenAIEmbeddings()
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="pharmacy_knowledge"
        )
    
    def setup_qa_chain(self, model_name: str = "gpt-4o", temperature: float = 0) -> None:
        """
        Set up the QA chain for question answering.
        Uses GitHub-hosted models.
        
        Args:
            model_name: Model to use (default: gpt-4o, GitHub-hosted)
            temperature: Temperature for response generation (0 = deterministic)
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        # Create LLM using GitHub-hosted models
        # Credentials are provided by the environment
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Create custom prompt template
        prompt_template = """You are a knowledgeable pharmacy assistant. Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context provided, say so - don't make up information.
Always cite which medication or topic your answer relates to.

Context:
{context}

Question: {question}

Helpful Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's pharmacy-related question
            
        Returns:
            Dictionary containing the answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call setup_qa_chain first.")
        
        # Get response
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def initialize(self, model_name: str = "gpt-4o") -> None:
        """
        Complete initialization of the RAG system.
        
        Args:
            model_name: Model to use for generation (default: gpt-4o, GitHub-hosted)
        """
        # Load and process documents
        documents = self.load_and_process_documents()
        
        # Create vector store
        self.create_vectorstore(documents)
        
        # Setup QA chain
        self.setup_qa_chain(model_name=model_name)


def create_rag_system(data_path: str = "data/pharmacy_info.txt", 
                      model_name: str = "gpt-4o") -> PharmacyRAG:
    """
    Convenience function to create and initialize a Pharmacy RAG system.
    Uses GitHub-hosted models - no API key required.
    
    Args:
        data_path: Path to pharmacy data file
        model_name: Model to use (default: gpt-4o, GitHub-hosted)
        
    Returns:
        Initialized PharmacyRAG instance
    """
    rag = PharmacyRAG(data_path=data_path)
    rag.initialize(model_name=model_name)
    return rag
