import numpy as np
from typing import List, Dict

class BM25Retriever:
    """BM25 baseline using rank_bm25."""
    def __init__(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank_bm25 (pip install rank_bm25)")
        self.BM25Okapi = BM25Okapi
        self.bm25 = None
        self._metadata = []

    def store(self, texts: List[str], metadata: List[Dict]):
        # Simple whitespace tokenization for baseline
        tokenized_corpus = [doc.lower().split() for doc in texts]
        self.bm25 = self.BM25Okapi(tokenized_corpus)
        self._metadata = metadata

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.bm25 is None:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_k = min(top_k, len(self._metadata))
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            meta = self._metadata[idx].copy()
            meta['similarity_score'] = scores[idx]
            meta['rank'] = rank + 1
            results.append(meta)
            
        return results

class DenseRetriever:
    """Standard Dense Retriever using Cosine Similarity (DPR baseline)."""
    def __init__(self):
        self._patterns = None
        self._metadata = []

    def store(self, patterns: np.ndarray, metadata: List[Dict]):
        norms = np.linalg.norm(patterns, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self._patterns = patterns / norms
        self._metadata = metadata

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self._patterns is None:
            return []
            
        q = query_embedding.copy()
        n = np.linalg.norm(q)
        if n > 1e-8:
            q = q / n
            
        similarities = self._patterns @ q
        
        top_k = min(top_k, len(self._metadata))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            meta = self._metadata[idx].copy()
            meta['similarity_score'] = float(similarities[idx])
            meta['rank'] = rank + 1
            results.append(meta)
            
        return results
