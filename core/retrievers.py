"""
Retriever Subsystem & Adapters
==============================
Defines the stable Retriever interface and concrete adapters:
1. HopfieldRetrieverAdapter (Continuous Attractor Dynamics)
2. DenseRetrieverAdapter (Static Cosine Similarity baseline)
3. HybridRetrieverAdapter (BM25 Lexical + Continuous Embedding Fusion via RRF)

All adapters return the unified EvidenceCollection containing EvidenceChunks.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from core.schemas import EvidenceChunk, EvidenceCollection
from core.baselines import BM25Retriever, DenseRetriever
from core.associative_engine import HopfieldAssociativeEngine

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract Base Class for all retrieval backends."""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity_threshold: float = 0.05
    ) -> EvidenceCollection:
        """Execute query and return standardized EvidenceCollection."""
        pass


class HopfieldRetrieverAdapter(BaseRetriever):
    """Adapter for Continuous Modern Hopfield Associative Memory Engine."""
    
    def __init__(self, associative_engine: HopfieldAssociativeEngine):
        self.engine = associative_engine

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity_threshold: float = 0.05
    ) -> EvidenceCollection:
        t0 = time.time()
        res = self.engine.retrieve_associative(
            query_embedding=query_embedding,
            raw_query=query,
            top_k=top_k,
            use_projection=getattr(self.engine.memory.retriever, 'use_projection', False)
        )
        self.last_result = res
        elapsed = (time.time() - t0) * 1000

        chunks: List[EvidenceChunk] = []
        for i, c in enumerate(res.all_retrieved_chunks):
            if c.similarity_score < min_similarity_threshold:
                continue
            meta = getattr(c, 'metadata', {}) or {}
            chunk = EvidenceChunk(
                chunk_id=c.chunk_id,
                doc_id=meta.get("doc_id", c.source),
                source=c.source,
                text=c.text,
                similarity_score=float(c.similarity_score),
                hopfield_attention=float(c.hopfield_attention),
                rank=i + 1,
                page_number=meta.get("page_number"),
                char_offset_start=meta.get("start_char"),
                char_offset_end=meta.get("end_char"),
                is_core_attractor=meta.get("is_core_attractor", True),
                metadata=meta
            )
            chunks.append(chunk)

        trajectory_data = [
            {
                "hop": h.hop_index,
                "energy": round(h.energy, 4),
                "delta": round(h.state_delta, 4),
                "description": h.description
            } for h in res.trajectory
        ]

        return EvidenceCollection(
            query=query,
            chunks=chunks,
            retriever_name="Hopfield Associative Memory",
            retrieval_time_ms=elapsed,
            attractor_energy=res.attractor_energy,
            energy_depth=res.energy_landscape_depth,
            is_grounded=res.is_grounded,
            trajectory_hops=trajectory_data
        )


class DenseRetrieverAdapter(BaseRetriever):
    """Adapter for Standard Dense Cosine Similarity (DPR / Vector DB baseline)."""
    
    def __init__(self, memory):
        self.memory = memory

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity_threshold: float = 0.05
    ) -> EvidenceCollection:
        t0 = time.time()
        if self.memory._patterns is None or self.memory.n_patterns == 0:
            return EvidenceCollection(
                query=query,
                chunks=[],
                retriever_name="Dense Cosine Similarity",
                retrieval_time_ms=0.0,
                is_grounded=False
            )

        patterns = self.memory._patterns.cpu().numpy()
        metadata = self.memory._metadata
        
        q = query_embedding.copy()
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-8:
            q = q / q_norm
            
        sims = patterns @ q # Cosine similarity
        top_k_actual = min(top_k, len(metadata))
        top_indices = np.argsort(sims)[::-1][:top_k_actual]
        elapsed = (time.time() - t0) * 1000

        chunks: List[EvidenceChunk] = []
        for rank, idx in enumerate(top_indices):
            score = float(sims[idx])
            if score < min_similarity_threshold:
                continue
            meta = metadata[idx]
            chunk = EvidenceChunk(
                chunk_id=meta.get("chunk_id", str(idx)),
                doc_id=meta.get("doc_id", meta.get("source", "unknown")),
                source=meta.get("source", "unknown"),
                text=meta.get("text", ""),
                similarity_score=score,
                hopfield_attention=0.0,
                rank=rank + 1,
                page_number=meta.get("page_number"),
                char_offset_start=meta.get("start_char"),
                char_offset_end=meta.get("end_char"),
                is_core_attractor=True,
                metadata=meta
            )
            chunks.append(chunk)

        return EvidenceCollection(
            query=query,
            chunks=chunks,
            retriever_name="Dense Cosine Similarity",
            retrieval_time_ms=elapsed,
            is_grounded=len(chunks) > 0
        )


class HybridRetrieverAdapter(BaseRetriever):
    """
    Hybrid Retriever combining BM25 Lexical Keyword search
    with Dense / Hopfield embeddings using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, dense_or_hopfield_adapter: BaseRetriever, memory):
        self.neural_retriever = dense_or_hopfield_adapter
        self.memory = memory
        self.bm25 = BM25Retriever()
        self._indexed_chunks_count = 0

    def _sync_bm25(self):
        """Index memory chunks into BM25 if new documents arrived."""
        if self.memory._metadata and len(self.memory._metadata) != self._indexed_chunks_count:
            texts = [m.get("text", "") for m in self.memory._metadata]
            self.bm25.store(texts, self.memory._metadata)
            self._indexed_chunks_count = len(self.memory._metadata)

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity_threshold: float = 0.05
    ) -> EvidenceCollection:
        t0 = time.time()
        self._sync_bm25()

        # 1. Retrieve neural candidates
        neural_collection = self.neural_retriever.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k * 2,
            min_similarity_threshold=min_similarity_threshold
        )

        # 2. Retrieve BM25 candidates
        bm25_raw = self.bm25.retrieve(query=query, top_k=top_k * 2)

        # 3. Reciprocal Rank Fusion (RRF)
        # Score = sum_m 1 / (60 + rank_m)
        rrf_k = 60.0
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, EvidenceChunk] = {}

        for rank, chunk in enumerate(neural_collection.chunks):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
            chunk_map[cid] = chunk

        for rank, b_meta in enumerate(bm25_raw):
            cid = b_meta.get("chunk_id", str(rank))
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
            if cid not in chunk_map:
                chunk_map[cid] = EvidenceChunk(
                    chunk_id=cid,
                    doc_id=b_meta.get("doc_id", b_meta.get("source", "unknown")),
                    source=b_meta.get("source", "unknown"),
                    text=b_meta.get("text", ""),
                    similarity_score=float(b_meta.get("similarity_score", 0.0)),
                    rank=rank + 1,
                    page_number=b_meta.get("page_number"),
                    metadata=b_meta
                )

        sorted_cids = sorted(scores.keys(), key=lambda c: -scores[c])[:top_k]
        elapsed = (time.time() - t0) * 1000

        fused_chunks = []
        for i, cid in enumerate(sorted_cids):
            c = chunk_map[cid]
            c.rank = i + 1
            fused_chunks.append(c)

        return EvidenceCollection(
            query=query,
            chunks=fused_chunks,
            retriever_name="Hybrid (BM25 + Neural RRF)",
            retrieval_time_ms=elapsed,
            attractor_energy=neural_collection.attractor_energy,
            energy_depth=neural_collection.energy_depth,
            is_grounded=len(fused_chunks) > 0,
            trajectory_hops=neural_collection.trajectory_hops
        )
