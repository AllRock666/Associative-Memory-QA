import torch
import pytest
from core.hopfield_torch import HopfieldRetriever

def test_custom_energy_functions():
    """
    Test that Hopfield Retriever can run with novel energy functions
    """
    torch.manual_seed(42)
    embedding_dim = 128
    num_patterns = 10
    
    query = torch.randn(1, 1, embedding_dim)
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    
    stored_patterns = torch.randn(num_patterns, embedding_dim)
    stored_patterns = torch.nn.functional.normalize(stored_patterns, p=2, dim=-1)
    
    # 1. Sparsemax
    retriever_sparse = HopfieldRetriever(beta=1.0, n_iter=1, energy_type="sparsemax")
    state_sparse, attn_sparse, _ = retriever_sparse(query, stored_patterns, top_k=5)
    
    # Assert sparsemax sums to 1
    assert torch.allclose(attn_sparse.sum(dim=-1), torch.ones_like(attn_sparse.sum(dim=-1)))
    
    # Assert it contains actual exact zeros (sparsity)
    assert (attn_sparse == 0.0).any().item(), "Sparsemax did not produce any zero weights"
    
    # 2. Polynomial
    retriever_poly = HopfieldRetriever(beta=2.0, n_iter=1, energy_type="polynomial")
    state_poly, attn_poly, _ = retriever_poly(query, stored_patterns, top_k=5)
    
    # Assert polynomial sums to 1
    assert torch.allclose(attn_poly.sum(dim=-1), torch.ones_like(attn_poly.sum(dim=-1)))
    
    print("Novel energy functions test passed!")

if __name__ == "__main__":
    pytest.main([__file__])
