import torch
import torch.nn.functional as F
import torch.optim as optim
from core.hopfield_torch import HopfieldRetriever

def test_projection_training():
    """
    Smoke test to prove that the QueryProjectionLayer can be trained 
    with a contrastive InfoNCE loss (gradients flow correctly).
    This does NOT represent actual performance, just architectural validity.
    """
    torch.manual_seed(42)
    embedding_dim = 64
    batch_size = 4
    num_patterns = 10
    
    # 1. Initialize model with projection enabled
    model = HopfieldRetriever(beta=1.0, n_iter=1, residual=True, use_projection=True, embedding_dim=embedding_dim)
    model.train()
    
    # 2. Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Create synthetic data
    queries = torch.randn(batch_size, embedding_dim)
    queries = F.normalize(queries, p=2, dim=-1)
    
    # Let's say the first 4 patterns in memory are the positive matches for the 4 queries
    stored_patterns = torch.randn(num_patterns, embedding_dim)
    stored_patterns[:batch_size] = queries + torch.randn(batch_size, embedding_dim) * 0.1
    stored_patterns = F.normalize(stored_patterns, p=2, dim=-1)
    
    # The gold standard indices for each query
    labels = torch.arange(batch_size)
    
    # 4. Training loop (just 5 steps)
    initial_loss = None
    final_loss = None
    
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward pass (get attention weights)
        _, attention, _ = model(queries, stored_patterns, top_k=num_patterns)
        attention = attention.squeeze(1) # shape: (B, N)
        
        # InfoNCE loss (maximize attention to the positive pattern)
        # log_softmax of the raw scaled projections is standard, but since HopfieldRetriever 
        # already does softmax inside, we can just use NLL on the attention probabilities.
        # However, for numerical stability, standard cross_entropy on logits is better.
        # But we only get attention from forward(). Let's compute manual NLL.
        # To avoid log(0), add epsilon
        log_probs = torch.log(attention + 1e-8)
        loss = F.nll_loss(log_probs, labels)
        
        if step == 0:
            initial_loss = loss.item()
            
        loss.backward()
        
        # Verify gradients are flowing through the projection layer
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item()
        assert grad_norm > 0, "Gradients are not flowing!"
        
        optimizer.step()
        final_loss = loss.item()
        
    print(f"Smoke test passed. Initial loss: {initial_loss:.4f} -> Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss did not decrease during synthetic training!"

if __name__ == "__main__":
    test_projection_training()
