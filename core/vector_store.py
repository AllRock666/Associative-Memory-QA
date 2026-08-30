import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """A single retrieval result from the Vector Store."""
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    similarity_score: float
    rank: int
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class VectorStore:
    """
    Standard Vector Store for exact K-Nearest Neighbor (KNN) search
    using Cosine Similarity on dense embeddings.
    """
    def __init__(self, similarity_threshold: float = 0.0):
        self.similarity_threshold = similarity_threshold
        self._patterns: Optional[np.ndarray] = None
        self._metadata: List[Dict] = []
        self._is_built = False

    @property
    def n_patterns(self) -> int:
        return len(self._metadata) if self._metadata else 0

    @property
    def embedding_dim(self) -> int:
        return self._patterns.shape[1] if self._patterns is not None else 0

    def store(self, patterns: np.ndarray, metadata: List[Dict]) -> None:
        """Store normalized memory patterns."""
        assert len(patterns) == len(metadata), "Patterns and metadata must match."
        
        # Normalize patterns for fast cosine similarity via dot product
        norms = np.linalg.norm(patterns, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self._patterns = patterns / norms
        self._metadata = metadata
        self._is_built = True

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve top-k most relevant patterns using cosine similarity."""
        if not self._is_built or self._patterns is None:
            return []

        # Normalize query
        query = query_embedding.copy()
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm

        # Cosine similarity via dot product
        similarities = self._patterns @ query

        # Get top-k indices
        top_k = min(top_k, self.n_patterns)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            sim_score = float(similarities[idx])
            if sim_score < self.similarity_threshold:
                continue

            meta = self._metadata[idx]
            results.append(RetrievalResult(
                chunk_id=meta.get("chunk_id", str(idx)),
                text=meta.get("text", ""),
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", idx),
                similarity_score=sim_score,
                rank=rank + 1,
                metadata=meta,
            ))

        return results

    def retrieve_with_fusion(
        self,
        query_embeddings: List[np.ndarray],
        top_k: int = 5,
        fusion_weights: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        """Multi-query fusion retrieval using Reciprocal Rank Fusion (RRF)."""
        if not query_embeddings:
            return []

        if fusion_weights is None:
            fusion_weights = [1.0] * len(query_embeddings)

        total = sum(fusion_weights)
        fusion_weights = [w / total for w in fusion_weights]

        all_results: Dict[str, Dict] = {}
        rrf_k = 60

        for q_emb, q_weight in zip(query_embeddings, fusion_weights):
            results = self.retrieve(q_emb, top_k=top_k * 2)

            for result in results:
                cid = result.chunk_id
                rrf_score = q_weight / (rrf_k + result.rank)

                if cid not in all_results:
                    all_results[cid] = {
                        "result": result,
                        "rrf_score": 0.0,
                        "max_similarity": result.similarity_score,
                    }
                all_results[cid]["rrf_score"] += rrf_score
                all_results[cid]["max_similarity"] = max(
                    all_results[cid]["max_similarity"], result.similarity_score
                )

        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:top_k]

        fused = []
        for rank, item in enumerate(sorted_results):
            r = item["result"]
            r.rank = rank + 1
            # We overwrite similarity with the max observed, and rank with the fused rank
            r.similarity_score = item["max_similarity"]
            # To maintain compatibility with some UI elements, we might still expose it as 'attention'
            # but we'll remove attention-specific references in the UI.
            fused.append(r)

        return fused

    def get_network_stats(self) -> Dict:
        """Return statistics about the stored vector database."""
        if not self._is_built:
            return {"status": "empty"}

        return {
            "n_patterns": self.n_patterns,
            "embedding_dim": self.embedding_dim,
            "storage_type": "Dense NumPy Vectors",
        }
