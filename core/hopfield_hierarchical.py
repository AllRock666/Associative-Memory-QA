"""
Hopfield Memory 2.0: Hierarchical Knowledge Memory
===================================================
Implements a 5-Level Multimodal Hierarchical Associative Memory Architecture:

Level 1: Raw Memory
    - Original raw elements, text, page numbers, coordinates, tables, and images.
Level 2: Semantic Memory
    - Dense 384-dim normalized embeddings + BM25Okapi sparse token index.
Level 3: Associative Memory
    - Continuous modern Hopfield network with exponential energy minimization (beta=8.0),
      multi-hop attractor trajectory tracking, and basin stability validation.
Level 4: Knowledge Graph Memory
    - Entity-relation triplets, concept links, cross-document associations, and neighborhood expansion.
Level 5: Episodic Memory
    - Query-answer logs, retrieved attractor IDs, user-confirmed corrections, and session stats.

Seamlessly integrates with PyTorchHopfieldMemory, HopfieldAssociativeEngine, and KnowledgeGraphEngine.
"""

import time
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field

import numpy as np
import torch
from rank_bm25 import BM25Okapi

from core.hopfield_torch import PyTorchHopfieldMemory, RetrievalResult
from core.associative_engine import HopfieldAssociativeEngine, AssociativeMemoryResult
from core.knowledge_graph import KnowledgeGraphEngine
from core.schemas import EvidenceChunk

logger = logging.getLogger(__name__)


@dataclass
class EpisodicEntry:
    """An interaction stored in Level 5 Episodic Memory."""
    session_id: str
    query: str
    retrieved_memory_ids: List[str]
    answer_summary: str
    user_feedback: Optional[str] = None  # e.g., "helpful", "corrected", "flagged"
    timestamp: float = field(default_factory=time.time)


@dataclass
class HierarchicalRetrievalResult:
    """Result of hierarchical associative retrieval across all 5 levels."""
    query: str
    core_attractors: List[Dict[str, Any]]
    associative_bridges: List[Dict[str, Any]]
    knowledge_graph_entities: List[Dict[str, Any]]
    energy: float
    energy_depth: float
    is_grounded: bool
    retrieval_time_ms: float
    episodic_context: List[Dict[str, Any]] = field(default_factory=list)
    multimodal_provenance: List[Dict[str, Any]] = field(default_factory=list)


class HierarchicalHopfieldMemory:
    """
    Unified 5-Level Hierarchical Knowledge Memory Orchestrator.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        beta: float = 8.0,
        device: str = "cpu",
        use_associative: bool = True
    ):
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.device = device
        self.use_associative = use_associative

        # Level 1: Raw Memory Store (keyed by chunk_id or memory_id)
        self.raw_memory: Dict[str, Dict[str, Any]] = {}
        self.content_hashes: Set[str] = set()

        # Level 2: Semantic Memory (Embeddings + BM25)
        self.hopfield_memory = PyTorchHopfieldMemory(
            embedding_dim=embedding_dim,
            beta=beta
        )
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus_ids: List[str] = []

        # Level 3: Associative Memory Engine
        self.associative_engine = HopfieldAssociativeEngine(
            memory=self.hopfield_memory,
            beta=beta,
            max_hops=3,
            alpha=0.7
        )

        # Level 4: Knowledge Graph Memory
        self.knowledge_graph = KnowledgeGraphEngine()

        # Level 5: Episodic Memory
        self.episodic_memory: List[EpisodicEntry] = []

    def ingest_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> int:
        """
        Ingests enriched chunks across all 5 memory levels.
        Deduplicates based on SHA-256 content hash.
        """
        new_chunks = []
        new_embeddings = []

        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            chash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if chash in self.content_hashes:
                continue  # Content-hash deduplication

            self.content_hashes.add(chash)
            cid = chunk.get("chunk_id") or f"mem_{len(self.raw_memory)}_{chash[:8]}"
            chunk["chunk_id"] = cid
            chunk["content_hash"] = chash

            # Level 1: Raw Memory
            self.raw_memory[cid] = chunk
            new_chunks.append(chunk)
            if idx < len(embeddings):
                new_embeddings.append(embeddings[idx])

        if not new_chunks:
            return 0

        # Level 2 & 3: Hopfield and BM25 index updates
        if new_embeddings:
            emb_matrix = np.array(new_embeddings, dtype=np.float32)
            self.hopfield_memory.add_patterns(emb_matrix, new_chunks)

        # Rebuild BM25 index over all raw memory items
        all_ids = list(self.raw_memory.keys())
        tokenized_corpus = [self.raw_memory[cid].get("text", "").lower().split() for cid in all_ids]
        if tokenized_corpus:
            self.bm25_index = BM25Okapi(tokenized_corpus)
            self.bm25_corpus_ids = all_ids

        # Level 4: Ingest into Knowledge Graph
        self.knowledge_graph.extract_and_ingest(new_chunks)

        return len(new_chunks)

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ) -> HierarchicalRetrievalResult:
        """
        Executes multi-level retrieval:
        1. BM25 sparse keyword ranking
        2. Continuous Hopfield associative relaxation
        3. Reciprocal Rank Fusion & co-attractor discovery
        4. Knowledge Graph neighborhood expansion
        5. Episodic context injection
        """
        t0 = time.time()

        # 1. Level 3 Continuous Hopfield Associative Relaxation
        assoc_result = self.associative_engine.retrieve_associative(
            query_embedding=query_embedding,
            raw_query=query,
            top_k=top_k
        )

        def _retrieval_to_dict(r) -> Dict[str, Any]:
            if hasattr(r, "to_dict"):
                return r.to_dict()
            return {
                "chunk_id": getattr(r, "chunk_id", ""),
                "text": getattr(r, "text", ""),
                "source": getattr(r, "source", ""),
                "similarity_score": getattr(r, "similarity_score", 0.0),
                "rank": getattr(r, "rank", 1),
                "metadata": getattr(r, "metadata", {}) or {},
                "hopfield_attention": getattr(r, "hopfield_attention", 0.0)
            }

        core_chunks = [_retrieval_to_dict(c) for c in assoc_result.core_attractor_chunks]
        bridge_chunks = [_retrieval_to_dict(c) for c in assoc_result.associative_bridge_chunks]

        # 2. Level 2 BM25 Sparse Ranking
        bm25_hits = []
        if self.bm25_index and self.bm25_corpus_ids:
            query_tokens = query.lower().split()
            scores = self.bm25_index.get_scores(query_tokens)
            top_bm25_indices = np.argsort(scores)[::-1][:top_k]
            for idx in top_bm25_indices:
                if scores[idx] > 0.05:
                    cid = self.bm25_corpus_ids[idx]
                    raw = self.raw_memory.get(cid, {})
                    bm25_hits.append({
                        "chunk_id": cid,
                        "text": raw.get("text", ""),
                        "source": raw.get("source", "Document"),
                        "page_number": raw.get("page_number", 1),
                        "element_type": raw.get("element_type", "text"),
                        "bm25_score": float(scores[idx])
                    })

        # Merge BM25 hits into bridges if not already in core attractors
        existing_ids = {c["chunk_id"] for c in core_chunks}
        for hit in bm25_hits:
            if hit["chunk_id"] not in existing_ids:
                bridge_chunks.append(hit)
                existing_ids.add(hit["chunk_id"])

        # 3. Level 4 Knowledge Graph Neighborhood Expansion
        seed_terms = query.split()
        kg_entities = self.knowledge_graph.expand_neighborhood(seed_terms, max_hops=2, max_nodes=10)

        # 4. Level 5 Episodic Context Lookups
        recent_episodic = []
        for ep in self.episodic_memory[-5:]:
            if any(term.lower() in ep.query.lower() for term in seed_terms if len(term) > 3):
                recent_episodic.append({
                    "query": ep.query,
                    "answer_summary": ep.answer_summary,
                    "feedback": ep.user_feedback
                })

        # 5. Multimodal Provenance
        provenance = []
        for c in (core_chunks + bridge_chunks)[:top_k]:
            cid = c.get("chunk_id")
            raw = self.raw_memory.get(cid, {})
            provenance.append({
                "chunk_id": cid,
                "source": c.get("source") or raw.get("source", "Document"),
                "page_number": c.get("page_number") or raw.get("page_number", 1),
                "element_type": c.get("element_type") or raw.get("element_type", "text"),
                "has_table": raw.get("element_type") == "table",
                "similarity_score": c.get("similarity_score", 0.8)
            })

        t_elapsed = (time.time() - t0) * 1000.0

        return HierarchicalRetrievalResult(
            query=query,
            core_attractors=core_chunks,
            associative_bridges=bridge_chunks[:top_k],
            knowledge_graph_entities=kg_entities,
            energy=assoc_result.attractor_energy,
            energy_depth=assoc_result.energy_landscape_depth,
            is_grounded=assoc_result.is_grounded,
            retrieval_time_ms=t_elapsed,
            episodic_context=recent_episodic,
            multimodal_provenance=provenance
        )

    def record_interaction(
        self,
        session_id: str,
        query: str,
        retrieved_memory_ids: List[str],
        answer_summary: str,
        user_feedback: Optional[str] = None
    ):
        """Records an interaction into Level 5 Episodic Memory."""
        self.episodic_memory.append(EpisodicEntry(
            session_id=session_id,
            query=query,
            retrieved_memory_ids=retrieved_memory_ids,
            answer_summary=answer_summary,
            user_feedback=user_feedback
        ))

    def get_stats(self) -> Dict[str, Any]:
        """Returns diagnostic and health statistics across all 5 memory layers."""
        return {
            "level1_raw_memory_count": len(self.raw_memory),
            "level2_semantic_patterns": self.hopfield_memory.n_patterns,
            "level3_associative_beta": self.beta,
            "level4_kg_nodes": self.knowledge_graph.graph.number_of_nodes(),
            "level4_kg_edges": self.knowledge_graph.graph.number_of_edges(),
            "level5_episodic_interactions": len(self.episodic_memory)
        }
