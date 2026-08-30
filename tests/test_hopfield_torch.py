import pytest
import torch
from core.hopfield_torch import HopfieldRetriever, BiLSTMQAHead

def test_hopfield_retriever():
    B, N, d = 2, 10, 64
    retriever = HopfieldRetriever(beta=1.0)
    
    query = torch.randn(B, d)
    memory = torch.randn(B, N, d)
    
    updated_state, attention, top_indices = retriever(query, memory, top_k=3)
    
    assert updated_state.shape == (B, 1, d)
    assert attention.shape == (B, 1, N)
    assert top_indices.shape == (B, 3)
    
    # Check probabilities sum to 1
    assert torch.allclose(attention.sum(dim=-1), torch.ones(B, 1))

def test_hopfield_retriever_unbatched_memory():
    B, N, d = 2, 10, 64
    retriever = HopfieldRetriever(beta=2.0)
    
    query = torch.randn(B, d)
    memory = torch.randn(N, d) # Unbatched memory
    
    updated_state, attention, top_indices = retriever(query, memory, top_k=3)
    assert updated_state.shape == (B, 1, d)
    assert attention.shape == (B, 1, N)
    assert top_indices.shape == (B, 3)

def test_bilstm_qa_head():
    B, L, input_dim = 2, 20, 128
    head = BiLSTMQAHead(input_dim=input_dim, hidden_dim=64)
    
    embeddings = torch.randn(B, L, input_dim)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, 15:] = False # simulate padding
    
    start_logits, end_logits = head(embeddings, mask)
    
    assert start_logits.shape == (B, L)
    assert end_logits.shape == (B, L)
    
    # Check masking
    assert (start_logits[:, 15:] == -1e4).all()
    assert (end_logits[:, 15:] == -1e4).all()

def test_bilstm_decode_span():
    head = BiLSTMQAHead(input_dim=128, hidden_dim=64)
    
    # Create fake logits where start is high at index 2, end is high at index 5
    start_logits = torch.randn(1, 10)
    end_logits = torch.randn(1, 10)
    
    start_logits[0, 2] = 100.0
    end_logits[0, 5] = 100.0
    
    best_span = head.decode_span(start_logits, end_logits)
    assert best_span == (2, 5)

import numpy as np
from core.hopfield_torch import PyTorchHopfieldMemory, RetrievalResult

def test_pytorch_hopfield_memory_wrapper_n0():
    memory = PyTorchHopfieldMemory()
    res = memory.retrieve(np.random.randn(64))
    assert len(res) == 0
    
    landscape = memory.get_attention_landscape(np.random.randn(64))
    assert landscape["attention_weights"] == []
    
    fusion = memory.retrieve_with_fusion([np.random.randn(64), np.random.randn(64)])
    assert len(fusion) == 0

def test_pytorch_hopfield_memory_wrapper_n1():
    memory = PyTorchHopfieldMemory(similarity_threshold=-1.0)
    patterns = np.random.randn(1, 64)
    metadata = [{"chunk_id": "test"}]
    memory.store(patterns, metadata)
    
    # Should not crash on squeeze
    res = memory.retrieve(np.random.randn(64), top_k=5)
    assert len(res) == 1
    assert res[0].chunk_id == "test"
    
    landscape = memory.get_attention_landscape(np.random.randn(64))
    assert len(landscape["attention_weights"]) == 1

def test_pytorch_hopfield_memory_wrapper_fusion():
    memory = PyTorchHopfieldMemory(similarity_threshold=-1.0)
    patterns = np.random.randn(10, 64)
    metadata = [{"chunk_id": f"test_{i}"} for i in range(10)]
    memory.store(patterns, metadata)
    
    queries = [np.random.randn(64) for _ in range(3)]
    res = memory.retrieve_with_fusion(queries, top_k=5, fusion_weights=[1.0, 0.5, 0.5])
    assert len(res) == 5
    assert isinstance(res[0], RetrievalResult)
