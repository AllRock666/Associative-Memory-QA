import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def sparsemax(z, dim=-1):
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    
    shape = [1] * z.dim()
    shape[dim] = z.size(dim)
    k = k.view(*shape)
    
    is_gt = z_sorted > (z_cumsum - 1) / k
    k_max = torch.sum(is_gt, dim=dim, keepdim=True)
    
    tau = (torch.gather(z_cumsum, dim, k_max - 1) - 1) / k_max
    return torch.relu(z - tau)

class QueryProjectionLayer(nn.Module):
    """
    Deep trainable projection layer with bottleneck expansion to separate query embeddings
    and provide a highly expressive landscape for the continuous Hopfield update.
    """
    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection
        return F.normalize(x + self.net(x), p=2, dim=-1)

class HopfieldRetriever(nn.Module):
    """
    Continuous Modern Hopfield Network Retriever.
    Computes the update rule iteratively: state_{t+1} = X * softmax(beta * X^T * state_t - M)
    """
    def __init__(self, beta: float = 1.0, n_iter: int = 1, residual: bool = True, use_projection: bool = False, embedding_dim: int = 384, energy_type: str = 'softmax'):
        super().__init__()
        self.beta = beta
        self.n_iter = n_iter
        self.residual = residual
        self.energy_type = energy_type
        
        # Untrained by default. PyTorchHopfieldMemory must opt-in to use it.
        self.use_projection = use_projection
        if use_projection:
            self.projection = QueryProjectionLayer(embedding_dim)
        else:
            self.projection = None

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
            if d != d_mem:
                raise ValueError("Dimension mismatch between query and memory")
            stored_patterns = stored_patterns.unsqueeze(0).expand(B, N, d)
            
        if self.use_projection and self.projection is not None:
            query = self.projection(query)
            
        state = query
        attention_weights = None
        
        # Iterative attractor dynamics
        for _ in range(self.n_iter):
            projection = torch.bmm(state, stored_patterns.transpose(1, 2))
            
            if self.energy_type == 'softmax':
                scaled_projection = self.beta * projection
                M, _ = torch.max(scaled_projection, dim=-1, keepdim=True)
                attention_weights = F.softmax(scaled_projection - M, dim=-1)
            elif self.energy_type == 'sparsemax':
                scaled_projection = self.beta * projection
                attention_weights = sparsemax(scaled_projection, dim=-1)
            elif self.energy_type == 'polynomial':
                # Polynomial energy: E = -sum( max(0, X^T xi)^beta )
                # Update rule uses activated projection
                degree = max(2.0, self.beta)
                activated = torch.relu(projection) ** degree
                attention_weights = activated / (activated.sum(dim=-1, keepdim=True) + 1e-8)
            else:
                raise ValueError(f"Unknown energy type: {self.energy_type}")
            
            retrieved = torch.bmm(attention_weights, stored_patterns)
            
            # Residual connection (Standard in Transformer/Modern Hopfield)
            if self.residual:
                state = F.normalize(query + retrieved, p=2, dim=-1)
            else:
                state = retrieved
            
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
        scores = start_logits.unsqueeze(1) + end_logits.unsqueeze(0)  # (L, L)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=scores.device))
        mask &= (torch.arange(L, device=scores.device).unsqueeze(1) + max_len > torch.arange(L, device=scores.device).unsqueeze(0))
        scores = scores.masked_fill(~mask, -float('inf'))
        best = torch.argmax(scores)
        return (best.item() // L, best.item() % L)

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
    def __init__(self, beta: float = 8.0, n_iter: int = 3, residual: bool = True, similarity_threshold: float = 0.05, use_projection: bool = False, embedding_dim: int = 384, energy_type: str = 'softmax'):
        self.retriever = HopfieldRetriever(beta=beta, n_iter=n_iter, residual=residual, use_projection=use_projection, embedding_dim=embedding_dim, energy_type=energy_type)
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
        self._metadata = list(metadata)

    def add_patterns(self, patterns: np.ndarray, metadata: List[Dict]) -> None:
        """Incrementally append new patterns and metadata without rebuilding existing memory."""
        if patterns is None or len(patterns) == 0:
            return
        norms = np.linalg.norm(patterns, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        patterns_norm = patterns / norms
        new_tensor = torch.tensor(patterns_norm, dtype=torch.float32, device=self.device)

        if self._patterns is None or self._patterns.shape[0] == 0:
            self._patterns = new_tensor
            self._metadata = list(metadata)
        else:
            self._patterns = torch.cat([self._patterns, new_tensor], dim=0)
            self._metadata.extend(metadata)

    def clear(self) -> None:
        """Clear all stored patterns and metadata."""
        self._patterns = None
        self._metadata = []

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        if self._patterns is None:
            return []
            
        q = query_embedding.copy()
        n = np.linalg.norm(q)
        if n > 1e-8:
            q = q / n
            
        q_tensor = torch.tensor(q, dtype=torch.float32, device=self.device).unsqueeze(0) # (1, d)
        
        raw_sims = torch.mm(q_tensor, self._patterns.T).squeeze(0).cpu().numpy() # (N,)
        
        with torch.no_grad():
            try:
                _, attn, top_idx = self.retriever(q_tensor, self._patterns, top_k=min(top_k, self.n_patterns))
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    self.retriever.to('cpu')
                    self._patterns = self._patterns.to('cpu')
                    q_tensor = q_tensor.to('cpu')
                    _, attn, top_idx = self.retriever(q_tensor, self._patterns, top_k=min(top_k, self.n_patterns))
                    self.device = torch.device('cpu')
                else:
                    raise e
            
        attn = attn.squeeze(0).squeeze(0).cpu().numpy()
        top_idx = top_idx.squeeze(0).squeeze(0).cpu().numpy()
        
        if top_idx.ndim == 0:
            top_idx = [int(top_idx)]
        else:
            top_idx = top_idx.tolist()
            
        results = []
        for rank, idx in enumerate(top_idx):
            idx = int(idx)
            attention = float(attn[idx])
            score = float(raw_sims[idx])
            if score < self.similarity_threshold:
                continue
            meta = self._metadata[idx]
            results.append(RetrievalResult(
                chunk_id=meta.get("chunk_id", str(idx)),
                text=meta.get("text", ""),
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", idx),
                similarity_score=score,
                hopfield_attention=attention,
                rank=rank+1,
                metadata=meta
            ))
        return results

    def retrieve_with_fusion(self, queries: List[np.ndarray], top_k: int = 5, fusion_weights: List[float] = None) -> List[RetrievalResult]:
        if self._patterns is None or not queries:
            return []
            
        B = len(queries)
        if fusion_weights is None:
            fusion_weights = [1.0] * B
            
        queries_arr = np.array(queries)
        norms = np.linalg.norm(queries_arr, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        q_norm = queries_arr / norms
        
        q_tensor = torch.tensor(q_norm, dtype=torch.float32, device=self.device)
        raw_sims = torch.mm(q_tensor, self._patterns.T).cpu().numpy()
        
        with torch.no_grad():
            try:
                _, attn, _ = self.retriever(q_tensor, self._patterns, top_k=min(top_k, self.n_patterns))
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    self.retriever.to('cpu')
                    self._patterns = self._patterns.to('cpu')
                    q_tensor = q_tensor.to('cpu')
                    _, attn, _ = self.retriever(q_tensor, self._patterns, top_k=min(top_k, self.n_patterns))
                    self.device = torch.device('cpu')
                else:
                    raise e
                    
        attn = attn.squeeze(1).cpu().numpy()
        
        chunk_scores = {}
        for b in range(B):
            w = fusion_weights[b]
            ranks = np.argsort(attn[b])[::-1]
            for rank_pos, idx in enumerate(ranks):
                rrf_score = w * (1.0 / (60 + rank_pos))
                if idx not in chunk_scores:
                    chunk_scores[idx] = {"rrf": 0.0, "max_attn": 0.0, "max_sim": 0.0}
                chunk_scores[idx]["rrf"] += rrf_score
                chunk_scores[idx]["max_attn"] = max(chunk_scores[idx]["max_attn"], float(attn[b, idx]))
                chunk_scores[idx]["max_sim"] = max(chunk_scores[idx]["max_sim"], float(raw_sims[b, idx]))
                
        sorted_indices = sorted(chunk_scores.keys(), key=lambda i: chunk_scores[i]["rrf"], reverse=True)
        
        results = []
        rank = 1
        for idx in sorted_indices:
            idx = int(idx)
            info = chunk_scores[idx]
            if info["max_sim"] < self.similarity_threshold:
                continue
            meta = self._metadata[idx]
            results.append(RetrievalResult(
                chunk_id=meta.get("chunk_id", str(idx)),
                text=meta.get("text", ""),
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", idx),
                similarity_score=info["max_sim"],
                hopfield_attention=info["max_attn"],
                rank=rank,
                metadata=meta
            ))
            rank += 1
            if len(results) >= top_k:
                break
                
        return results

    def get_attention_landscape(self, query: np.ndarray) -> Dict:
        if self._patterns is None:
            return {"attention_weights": [], "energy_values": []}
            
        q = query.copy()
        n = np.linalg.norm(q)
        if n > 1e-8:
            q = q / n
            
        q_tensor = torch.tensor(q, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            projection = torch.mm(q_tensor, self._patterns.T).squeeze(0)
            # Energy contribution of each pattern is inversely related to projection
            energy_values = -self.beta * projection
            _, attn, _ = self.retriever(q_tensor, self._patterns, top_k=1)
            
        attn_list = attn.squeeze(0).squeeze(0).cpu().numpy().tolist()
        if not isinstance(attn_list, list):
            attn_list = [attn_list]
            
        return {
            "attention_weights": attn_list,
            "energy_values": energy_values.cpu().numpy().tolist()
        }

    def get_network_stats(self) -> Dict:
        return {
            "n_patterns": self.n_patterns,
            "embedding_dim": self.embedding_dim,
            "theoretical_capacity": "PyTorch Dense"
        }
