import pytest
import numpy as np
from core.vector_store import VectorStore

def test_vector_store_init():
    store = VectorStore()
    assert store.similarity_threshold == 0.0
    assert store.n_patterns == 0
    assert store.embedding_dim == 0

def test_vector_store_store_and_retrieve():
    store = VectorStore()
    
    # Create dummy embeddings (d=4)
    patterns = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    metadata = [
        {"chunk_id": "1", "text": "doc1"},
        {"chunk_id": "2", "text": "doc2"},
        {"chunk_id": "3", "text": "doc3"}
    ]
    
    store.store(patterns, metadata)
    assert store.n_patterns == 3
    assert store.embedding_dim == 4
    
    # Query matching the first pattern closely
    query = np.array([0.9, 0.1, 0.0, 0.0])
    results = store.retrieve(query, top_k=2)
    
    assert len(results) == 2
    assert results[0].chunk_id == "1"
    assert results[0].text == "doc1"

def test_vector_store_fusion():
    store = VectorStore()
    patterns = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadata = [{"chunk_id": "1"}, {"chunk_id": "2"}]
    store.store(patterns, metadata)
    
    queries = [np.array([0.9, 0.1]), np.array([0.8, 0.2])]
    results = store.retrieve_with_fusion(queries, top_k=1)
    
    assert len(results) == 1
    assert results[0].chunk_id == "1"
