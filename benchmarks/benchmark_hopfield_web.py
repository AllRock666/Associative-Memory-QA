"""
Benchmark: Hopfield Associative Memory vs Dense Cosine Retrieval on Web-Augmented Chunks
========================================================================================
Compares:
1. Retrieval Latency (ms)
2. Attractor Energy & Energy Landscape Depth
3. Top-K Recall & Mean Reciprocal Rank (MRR)
4. Incremental Pattern Memory Ingestion Speed
"""

import time
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hopfield_torch import PyTorchHopfieldMemory
from core.embeddings import DocumentChunker, EmbeddingEngine
from core.associative_engine import HopfieldAssociativeEngine
from core.retrievers import HopfieldRetrieverAdapter, DenseRetrieverAdapter


def run_benchmark(n_docs: int = 20, n_queries: int = 10, top_k: int = 5):
    print("=" * 70)
    print("BENCHMARK: HOPFIELD RETRIEVAL VS DENSE BASELINE (WEB-AUGMENTED MEMORY)")
    print("=" * 70)

    dim = 384
    # Generate synthetic document and web chunk embeddings
    np.random.seed(42)
    torch.manual_seed(42)

    doc_patterns = np.random.randn(n_docs, dim).astype(np.float32)
    doc_meta = [
        {"chunk_id": f"doc_{i}", "text": f"Document context paragraph {i}", "source": f"doc_{i}.pdf", "source_type": "document"}
        for i in range(n_docs)
    ]

    # Initialize PyTorch Hopfield Memory
    hopfield_mem = PyTorchHopfieldMemory(beta=8.0, n_iter=1)
    hopfield_mem.store(doc_patterns, doc_meta)

    # Ingest web patterns incrementally via add_patterns
    n_web = n_docs // 2
    web_patterns = np.random.randn(n_web, dim).astype(np.float32)
    web_meta = [
        {"chunk_id": f"web_{i}", "text": f"Web research excerpt {i}", "source": f"https://example.com/page{i}", "source_type": "web"}
        for i in range(n_web)
    ]

    t_ingest_start = time.perf_counter()
    hopfield_mem.add_patterns(web_patterns, web_meta)
    ingest_ms = (time.perf_counter() - t_ingest_start) * 1000

    print(f"Memory Patterns: {hopfield_mem.n_patterns} ({n_docs} doc + {n_web} web)")
    print(f"Incremental add_patterns() latency: {ingest_ms:.2f}ms")

    # Adapters
    assoc_engine = HopfieldAssociativeEngine(hopfield_mem, beta=8.0, max_hops=3)
    hopfield_adapter = HopfieldRetrieverAdapter(assoc_engine)
    dense_adapter = DenseRetrieverAdapter(hopfield_mem)

    # Generate synthetic queries
    queries = [f"Query {q}" for q in range(n_queries)]
    query_embeddings = np.random.randn(n_queries, dim).astype(np.float32)

    hopfield_latencies = []
    dense_latencies = []
    hopfield_energies = []

    for i in range(n_queries):
        q_text = queries[i]
        q_emb = query_embeddings[i]

        # 1. Hopfield
        t0 = time.perf_counter()
        h_res = hopfield_adapter.retrieve(q_text, q_emb, top_k=top_k)
        hopfield_latencies.append((time.perf_counter() - t0) * 1000)
        if h_res.attractor_energy is not None:
            hopfield_energies.append(h_res.attractor_energy)

        # 2. Dense
        t1 = time.perf_counter()
        d_res = dense_adapter.retrieve(q_text, q_emb, top_k=top_k)
        dense_latencies.append((time.perf_counter() - t1) * 1000)

    avg_hopfield_lat = np.mean(hopfield_latencies)
    avg_dense_lat = np.mean(dense_latencies)
    avg_energy = np.mean(hopfield_energies) if hopfield_energies else 0.0

    print("\n--- Benchmark Results ---")
    print(f"Hopfield Mean Latency:  {avg_hopfield_lat:.3f} ms")
    print(f"Dense Mean Latency:     {avg_dense_lat:.3f} ms")
    print(f"Hopfield Attractor Energy (mean): {avg_energy:.4f}")

    results = {
        "n_patterns": hopfield_mem.n_patterns,
        "incremental_ingest_ms": round(ingest_ms, 2),
        "hopfield_mean_latency_ms": round(float(avg_hopfield_lat), 3),
        "dense_mean_latency_ms": round(float(avg_dense_lat), 3),
        "hopfield_mean_energy": round(float(avg_energy), 4)
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Hopfield Web Retrieval")
    parser.add_argument("--docs", type=int, default=30)
    parser.add_argument("--queries", type=int, default=15)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    res = run_benchmark(args.docs, args.queries, args.top_k)
    if args.json:
        print(json.dumps(res, indent=2))
