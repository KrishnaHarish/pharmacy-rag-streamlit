#!/usr/bin/env python3
"""
Test script for the Pharmacy RAG application.
Tests basic functionality without requiring an OpenAI API key.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import streamlit
        import langchain
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_community.vectorstores import Chroma
        from langchain_community.document_loaders import TextLoader
        from rag_utils import PharmacyRAG, create_rag_system
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_document_loading():
    """Test document loading and processing."""
    print("\nTesting document loading...")
    try:
        from rag_utils import PharmacyRAG
        
        rag = PharmacyRAG(data_path='data/pharmacy_info.txt')
        docs = rag.load_and_process_documents()
        
        if len(docs) > 0:
            print(f"✓ Loaded and processed {len(docs)} document chunks")
            print(f"  First chunk preview: {docs[0].page_content[:100]}...")
            return True
        else:
            print("✗ No documents loaded")
            return False
    except Exception as e:
        print(f"✗ Document loading failed: {e}")
        return False

def test_data_file():
    """Test that the pharmacy data file exists and has content."""
    print("\nTesting data file...")
    try:
        data_path = 'data/pharmacy_info.txt'
        if not os.path.exists(data_path):
            print(f"✗ Data file not found: {data_path}")
            return False
        
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content) > 0:
            print(f"✓ Data file exists with {len(content)} characters")
            print(f"  Contains information about medications")
            return True
        else:
            print("✗ Data file is empty")
            return False
    except Exception as e:
        print(f"✗ Data file test failed: {e}")
        return False

def test_app_syntax():
    """Test that the Streamlit app has valid syntax."""
    print("\nTesting application syntax...")
    try:
        import py_compile
        py_compile.compile('app.py', doraise=True)
        py_compile.compile('rag_utils.py', doraise=True)
        print("✓ All Python files have valid syntax")
        return True
    except Exception as e:
        print(f"✗ Syntax check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Pharmacy RAG Application - Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_document_loading,
        test_data_file,
        test_app_syntax,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed!")
        print("\nTo run the application:")
        print("1. Ensure you're in a GitHub-authenticated environment (e.g., GitHub Codespaces)")
        print("2. Run: streamlit run app.py")
        print("3. The app uses GitHub-hosted models - no API key configuration needed")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
