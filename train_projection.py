import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from core.embeddings import EmbeddingEngine
from core.hopfield_torch import HopfieldRetriever

def train_projection(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # 1. Load dataset (train split, small subset for speed)
    dataset = load_dataset("squad_v2", split="train")
    dataset = dataset.shuffle(seed=42).select(range(args.train_size))
    
    # 2. Extract unique contexts and embed them
    print("Embedding contexts...")
    embedder = EmbeddingEngine()
    contexts = list(set([item['context'] for item in dataset]))
    
    context_to_id = {ctx: i for i, ctx in enumerate(contexts)}
    patterns = []
    
    # Batch embedding for speed
    batch_size = 32
    for i in tqdm(range(0, len(contexts), batch_size)):
        batch = contexts[i:i+batch_size]
        batch_embs = [embedder.embed_query(ctx) for ctx in batch]
        patterns.extend(batch_embs)
        
    patterns_tensor = torch.tensor(np.array(patterns), dtype=torch.float32).to(device)
    
    # 3. Setup model and optimizer
    embedding_dim = patterns_tensor.shape[-1]
    model = HopfieldRetriever(
        beta=args.beta, 
        n_iter=args.n_iter, 
        residual=True, 
        use_projection=True, 
        embedding_dim=embedding_dim,
        energy_type=args.energy_type
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 4. Prepare training pairs
    queries = []
    labels = []
    for item in dataset:
        q_emb = embedder.embed_query(item['question'])
        queries.append(q_emb)
        labels.append(context_to_id[item['context']])
        
    queries_tensor = torch.tensor(np.array(queries), dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    
    # 5. Training loop
    epochs = args.epochs
    train_batch_size = args.batch_size
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        permutation = torch.randperm(queries_tensor.size()[0])
        
        for i in tqdm(range(0, queries_tensor.size()[0], train_batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            indices = permutation[i:i+train_batch_size]
            batch_queries = queries_tensor[indices]
            batch_labels = labels_tensor[indices]
            
            # Forward pass: compute attention over all stored patterns
            _, attention, _ = model(batch_queries, patterns_tensor, top_k=patterns_tensor.size(0))
            attention = attention.squeeze(1) # Shape: (B, N)
            
            # Since attention is already soft/sparse-maxed, taking -log(att[label]) would work, 
            # but standard InfoNCE uses raw logits before softmax.
            # To get raw logits without modifying HopfieldRetriever too much, we can just 
            # use a simpler approach: MSE between attention and one-hot, or directly grab projections.
            
            # For a proper InfoNCE loss, we use the dot products of the projected query and the patterns
            projected_queries = model.projection(batch_queries)
            
            # Compute similarities
            # (B, dim) x (dim, N) -> (B, N)
            logits = torch.matmul(projected_queries, patterns_tensor.T) * model.beta
            
            loss = F.cross_entropy(logits, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / (len(queries_tensor)//train_batch_size + 1):.4f}")
        
    # 6. Save projection weights
    os.makedirs("models", exist_ok=True)
    save_path = "models/projection_squad.pt"
    torch.save(model.projection.state_dict(), save_path)
    print(f"Saved trained projection layer to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--energy_type", type=str, default="softmax")
    args = parser.parse_args()
    
    train_projection(args)
