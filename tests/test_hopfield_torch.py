import torch
import numpy as np
import pytest
from core.hopfield_torch import HopfieldRetriever

def test_n_iter_1_equivalence():
    """
    Test that Hopfield update with n_iter=1 produces identical ranking 
    (attention weights) to simple cosine similarity.
    """
    torch.manual_seed(42)
    embedding_dim = 128
    num_patterns = 50
    beta = 1.0
    
    query = torch.randn(1, 1, embedding_dim)
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    
    stored_patterns = torch.randn(num_patterns, embedding_dim)
    stored_patterns = torch.nn.functional.normalize(stored_patterns, p=2, dim=-1)
    
    # 1. Cosine similarity
    # query is (1, 1, D), stored_patterns is (N, D)
    # Cosine sim is just dot product since they are normalized
    cos_sim = torch.matmul(query.squeeze(1), stored_patterns.t()).squeeze(0)
    _, cos_top_indices = torch.topk(cos_sim, k=10)
    
    # 2. Hopfield Retriever with n_iter = 1
    # Note: residual=True doesn't affect attention calculation at step 1
    retriever = HopfieldRetriever(beta=beta, n_iter=1, residual=True, use_projection=False)
    
    with torch.no_grad():
        _, attention, hop_top_indices = retriever(query, stored_patterns, top_k=10)
        
    attention = attention.squeeze(0).squeeze(0)
    hop_top_indices = hop_top_indices.squeeze(0).squeeze(0)
    
    # Mathematical assertion: softmax is monotonic, so ranks should match exactly
    assert torch.equal(cos_top_indices, hop_top_indices), "n_iter=1 Hopfield does not rank identically to cosine similarity"
    
    # We can also verify that attention is exactly softmax(beta * cos_sim)
    expected_attention = torch.nn.functional.softmax(beta * cos_sim, dim=-1)
    assert torch.allclose(attention, expected_attention, atol=1e-6), "n_iter=1 attention weights do not match expected softmax"

if __name__ == "__main__":
    pytest.main([__file__])
