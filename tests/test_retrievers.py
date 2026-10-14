"""
Test Suite: Retriever Adapters and BaseRetriever Interface Contract
===================================================================
Verifies:
1. Stable interface for BaseRetriever across:
   - HopfieldRetrieverAdapter
   - DenseRetrieverAdapter
   - HybridRetrieverAdapter
2. Uniform return schema (EvidenceCollection, EvidenceChunk).
3. Ranking, score preservation, and metadata propagation.
"""

import pytest
import numpy as np
import torch

from core.schemas import EvidenceChunk, EvidenceCollection
from core.retrievers import BaseRetriever, HopfieldRetrieverAdapter, DenseRetrieverAdapter, HybridRetrieverAdapter
from core.hopfield_torch import PyTorchHopfieldMemory
from core.associative_engine import HopfieldAssociativeEngine


@pytest.fixture
def memory_with_sample_patterns():
    memory = PyTorchHopfieldMemory(beta=8.0, n_iter=2, similarity_threshold=0.01)
    
    # Create 3 deterministic embedding patterns of dim 8
    dim = 8
    p1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    p2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    p3 = np.array([0.707, 0.707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    patterns = np.vstack([p1, p2, p3])

    metadata = [
        {"chunk_id": "c1", "source": "ai_primer.txt", "text": "Artificial Intelligence encompasses machine learning and reasoning."},
        {"chunk_id": "c2", "source": "hopfield.txt", "text": "Hopfield networks store continuous memories as energy attractors."},
        {"chunk_id": "c3", "source": "hybrid.txt", "text": "Modern Hopfield networks use softmax energy for exponential memory capacity."}
    ]

    memory.store(patterns, metadata)
    return memory


def test_retriever_subsystem_adapters(memory_with_sample_patterns):
    memory = memory_with_sample_patterns
    associative_engine = HopfieldAssociativeEngine(memory=memory, beta=8.0, max_hops=3)

    adapters = [
        HopfieldRetrieverAdapter(associative_engine),
        DenseRetrieverAdapter(memory),
        HybridRetrieverAdapter(HopfieldRetrieverAdapter(associative_engine), memory)
    ]

    query_emb = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    query_str = "What is Artificial Intelligence?"

    for adapter in adapters:
        assert isinstance(adapter, BaseRetriever)

        collection = adapter.retrieve(
            query=query_str,
            query_embedding=query_emb,
            top_k=2,
            min_similarity_threshold=0.01
        )

        # Verify contract
        assert isinstance(collection, EvidenceCollection)
        assert collection.query == query_str
        assert len(collection.chunks) > 0
        assert collection.retriever_name is not None
        assert collection.retrieval_time_ms >= 0.0

        # Verify chunk structure
        for rank, chunk in enumerate(collection.chunks, 1):
            assert isinstance(chunk, EvidenceChunk)
            assert chunk.chunk_id in {"c1", "c2", "c3"}
            assert chunk.rank == rank
            assert chunk.text != ""
            assert chunk.similarity_score >= 0.0
            assert chunk.source in collection.sources

        # Check properties
        assert collection.top_score >= 0.0
        assert collection.is_empty is False
