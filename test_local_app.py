#!/usr/bin/env python3
"""
Test script for the local Pharmacy RAG application.
Tests basic functionality without downloading the full model.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import streamlit
        import pandas
        import numpy
        import requests
        from sentence_transformers import SentenceTransformer
        import faiss
        from llama_cpp import Llama
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_drugs_csv():
    """Test that drugs.csv exists and has correct structure."""
    print("\nTesting drugs.csv...")
    try:
        import pandas as pd
        
        if not os.path.exists('drugs.csv'):
            print("✗ drugs.csv not found")
            return False
        
        df = pd.read_csv('drugs.csv')
        
        if 'drug' not in df.columns or 'description' not in df.columns:
            print("✗ drugs.csv missing required columns")
            return False
        
        print(f"✓ drugs.csv loaded with {len(df)} entries")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample drugs: {', '.join(df['drug'].head(3).tolist())}")
        return True
        
    except Exception as e:
        print(f"✗ drugs.csv test failed: {e}")
        return False

def test_embeddings():
    """Test sentence-transformers embedding model."""
    print("\nTesting sentence-transformers...")
    print("  Note: First run will download the model (~90MB)")
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test encoding
        test_text = ["Aspirin is a pain reliever"]
        embeddings = model.encode(test_text)
        
        print(f"✓ Embeddings model loaded")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        return True
        
    except Exception as e:
        print(f"✗ Embeddings test failed: {e}")
        return False

def test_faiss():
    """Test FAISS index creation."""
    print("\nTesting FAISS...")
    try:
        import faiss
        import numpy as np
        
        # Create simple index
        dimension = 384  # all-MiniLM-L6-v2 dimension
        index = faiss.IndexFlatIP(dimension)
        
        # Add some vectors
        vectors = np.random.random((10, dimension)).astype('float32')
        index.add(vectors)
        
        # Search
        query = np.random.random((1, dimension)).astype('float32')
        scores, indices = index.search(query, 3)
        
        print(f"✓ FAISS index created and tested")
        print(f"  Index size: {index.ntotal} vectors")
        return True
        
    except Exception as e:
        print(f"✗ FAISS test failed: {e}")
        return False

def test_app_syntax():
    """Test that app.py has valid syntax."""
    print("\nTesting app.py syntax...")
    try:
        import py_compile
        py_compile.compile('app.py', doraise=True)
        print("✓ app.py has valid syntax")
        return True
    except Exception as e:
        print(f"✗ Syntax check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Local Pharmacy RAG Application - Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_drugs_csv,
        test_embeddings,
        test_faiss,
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
        print("1. Run: streamlit run app.py")
        print("2. The app will download the TinyLlama model on first run (~640MB)")
        print("3. Subsequent runs will load from cache")
        print("\nNote: Model download may take a few minutes on first run")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
