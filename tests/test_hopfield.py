import pytest
import numpy as np
from core.hopfield import ModernHopfieldNetwork

def test_hopfield_network_init():
    network = ModernHopfieldNetwork(beta=8.0)
    assert network.beta == 8.0
    assert network.n_patterns == 0
    assert network.theoretical_capacity == 0

def test_hopfield_network_store_and_retrieve():
    network = ModernHopfieldNetwork(beta=8.0)
    
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
    
    network.store(patterns, metadata)
    assert network.n_patterns == 3
    assert network.embedding_dim == 4
    
    # Query matching the first pattern
    query = np.array([0.9, 0.1, 0.0, 0.0])
    results = network.retrieve(query, top_k=2)
    
    assert len(results) == 2
    assert results[0].chunk_id == "1"
    assert results[0].text == "doc1"
    
def test_hopfield_energy():
    network = ModernHopfieldNetwork(beta=8.0)
    patterns = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadata = [{"chunk_id": "1"}, {"chunk_id": "2"}]
    network.store(patterns, metadata)
    
    query = np.array([1.0, 0.0])
    energy = network._hopfield_energy(query)
    
    assert isinstance(energy, float)
    
def test_hopfield_fusion():
    network = ModernHopfieldNetwork(beta=8.0)
    patterns = np.array([[1.0, 0.0], [0.0, 1.0]])
    metadata = [{"chunk_id": "1"}, {"chunk_id": "2"}]
    network.store(patterns, metadata)
    
    queries = [np.array([0.9, 0.1]), np.array([0.8, 0.2])]
    results = network.retrieve_with_fusion(queries, top_k=1)
    
    assert len(results) == 1
    assert results[0].chunk_id == "1"
