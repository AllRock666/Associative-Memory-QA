import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class HopfieldRetriever(nn.Module):
    """
    Continuous Modern Hopfield Network Retriever.
    Computes the update rule iteratively: state_{t+1} = X * softmax(beta * X^T * state_t - M)
    """
    def __init__(self, beta: float = 1.0, n_iter: int = 1):
        super().__init__()
        self.beta = beta
        self.n_iter = n_iter

    def forward(
        self, 
        query: torch.Tensor, 
        stored_patterns: torch.Tensor, 
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.dim() == 2:
            query = query.unsqueeze(1)
        B, _, d = query.shape
            
        if stored_patterns.dim() == 2:
            N, d_mem = stored_patterns.shape
            assert d == d_mem, "Dimension mismatch between query and memory"
            stored_patterns = stored_patterns.unsqueeze(0).expand(B, N, d)
            
        state = query
        attention_weights = None
        
        # Iterative attractor dynamics
        for _ in range(self.n_iter):
            projection = torch.bmm(state, stored_patterns.transpose(1, 2))
            scaled_projection = self.beta * projection
            M, _ = torch.max(scaled_projection, dim=-1, keepdim=True)
            attention_weights = F.softmax(scaled_projection - M, dim=-1)
            state = torch.bmm(attention_weights, stored_patterns)
            
        _, top_indices = torch.topk(attention_weights.squeeze(1), k=min(top_k, attention_weights.size(-1)), dim=-1)
        
        return state, attention_weights, top_indices


class BiLSTMQAHead(nn.Module):
    """
    Extractive QA Head using a BiLSTM and MLP.
    Takes token embeddings, runs BiLSTM, and predicts start/end logits.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # BiLSTM outputs 2 * hidden_dim
        self.start_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.end_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, token_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_embeddings: Shape (B, L, input_dim)
            mask: Boolean tensor of shape (B, L) where True indicates valid tokens.
            
        Returns:
            start_logits: Shape (B, L)
            end_logits: Shape (B, L)
        """
        # BiLSTM pass
        # output shape: (B, L, 2 * hidden_dim)
        lstm_out, _ = self.bilstm(token_embeddings)
        
        # MLP passes
        # logits shape: (B, L, 1) -> (B, L)
        start_logits = self.start_mlp(lstm_out).squeeze(-1)
        end_logits = self.end_mlp(lstm_out).squeeze(-1)
        
        if mask is not None:
            # Mask out invalid padding tokens by setting logits to large negative value
            start_logits = start_logits.masked_fill(~mask, -1e4)
            end_logits = end_logits.masked_fill(~mask, -1e4)
            
        return start_logits, end_logits

    @torch.no_grad()
    def decode_span(self, start_logits: torch.Tensor, end_logits: torch.Tensor, max_len: int = 30) -> Tuple[int, int]:
        """
        Decodes the best valid span from the logits for a single sequence.
        Assumes batch size of 1.
        """
        start_logits = start_logits.squeeze(0)
        end_logits = end_logits.squeeze(0)
        
        L = start_logits.size(0)
        best_score = -float('inf')
        best_span = (0, 0)
        
        # Simple O(L^2) search for the best valid span
        for i in range(L):
            for j in range(i, min(i + max_len, L)):
                score = start_logits[i].item() + end_logits[j].item()
                if score > best_score:
                    best_score = score
                    best_span = (i, j)
                    
        return best_span

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    similarity_score: float
    rank: int
    metadata: Dict = None
    hopfield_attention: float = 0.0

class PyTorchHopfieldMemory:
    """Wrapper to mimic the old ModernHopfieldNetwork API but backed by PyTorch."""
    def __init__(self, beta: float = 8.0, n_iter: int = 3, similarity_threshold: float = 0.05):
        self.retriever = HopfieldRetriever(beta=beta, n_iter=n_iter)
        self.similarity_threshold = similarity_threshold
        self._patterns = None
        self._metadata = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.retriever.to(self.device)
        self.beta = beta

    @property
    def n_patterns(self) -> int:
        return len(self._metadata)

    @property
    def embedding_dim(self) -> int:
        return self._patterns.shape[1] if self._patterns is not None else 0

    def store(self, patterns: np.ndarray, metadata: List[Dict]) -> None:
        # Normalize patterns
        norms = np.linalg.norm(patterns, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        patterns_norm = patterns / norms
        self._patterns = torch.tensor(patterns_norm, dtype=torch.float32, device=self.device)
        self._metadata = metadata

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        if self._patterns is None:
            return []
            
        q = query_embedding.copy()
        n = np.linalg.norm(q)
        if n > 1e-8:
            q = q / n
            
        q_tensor = torch.tensor(q, dtype=torch.float32, device=self.device).unsqueeze(0) # (1, d)
        
        with torch.no_grad():
            _, attn, top_idx = self.retriever(q_tensor, self._patterns, top_k=min(top_k, self.n_patterns))
            
        attn = attn.squeeze().cpu().numpy()
        top_idx = top_idx.squeeze().cpu().numpy()
        
        if top_idx.ndim == 0:
            top_idx = [int(top_idx)]
        else:
            top_idx = top_idx.tolist()
            
        results = []
        for rank, idx in enumerate(top_idx):
            score = float(attn[idx])
            if score < self.similarity_threshold:
                continue
            meta = self._metadata[idx]
            results.append(RetrievalResult(
                chunk_id=meta.get("chunk_id", str(idx)),
                text=meta.get("text", ""),
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", idx),
                similarity_score=score,
                hopfield_attention=score,
                rank=rank+1,
                metadata=meta
            ))
        return results

    def retrieve_with_fusion(self, queries: List[np.ndarray], top_k: int = 5, fusion_weights: List[float] = None) -> List[RetrievalResult]:
        return self.retrieve(queries[0], top_k)

    def get_attention_landscape(self, query: np.ndarray) -> Dict:
        if self._patterns is None:
            return {"attention_weights": [], "energy_values": []}
        q_tensor = torch.tensor(query, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, attn, _ = self.retriever(q_tensor, self._patterns, top_k=1)
        attn_list = attn.squeeze().cpu().numpy().tolist()
        return {
            "attention_weights": attn_list,
            "energy_values": [0.0] * len(attn_list)
        }

    def get_network_stats(self) -> Dict:
        return {
            "n_patterns": self.n_patterns,
            "embedding_dim": self.embedding_dim,
            "theoretical_capacity": "PyTorch Dense"
        }
