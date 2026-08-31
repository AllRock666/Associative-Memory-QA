# 🧠 Associative Memory Retriever for Long-Context QA
### Exploring Modern Hopfield Networks for Retrieval

**Notice: This project is currently a work in progress and is under active development. It serves as a clean, testable reference implementation for exploring the equivalence between Modern Hopfield Networks and Transformer attention, rather than a production replacement for RAG.**

> An experimental QA system investigating energy-minimization-based associative memory retrieval — including when iterative attractor dynamics can overcome noise and when they collapse.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  DOCUMENT INGESTION                                             │
│  Text → DocumentChunker → EmbeddingEngine → HopfieldMemory     │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  QUERY PROCESSING                                               │
│  RawQuery → Clean → NoiseDetect → Expand → Variants           │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  HOPFIELD RETRIEVAL                                             │
│  QueryEmbedding → EnergyMinimization → SoftmaxAttention → Top-K│
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  ANSWER GENERATION                                              │
│  Chunks → Extractive QA Model (RoBERTa) → Span + Confidence     │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 The Math: Modern Hopfield Networks

### Energy Function
```
E = −lse(β, X^T ξ) + ½ξ^Tξ + (1/β)log N + ½M²
```

### Update Rule
```
ξ_new = X · softmax(β · X^T · ξ)
```
*Note on equivalence: With `n_iter=1`, ranking chunks by the Hopfield attention weight and ranking them by raw cosine similarity (Dense retrieval) produce the exact same ordering, as softmax is a strictly monotonic transform of similarity scores for a fixed β.*

## 📊 Evaluation & Benchmarks

We systematically evaluate the retrieval architecture to identify when Hopfield dynamics provide an advantage.

### SQuAD (Single-Hop Fact Retrieval)
On standard single-hop retrieval, a 1-step Hopfield update is mathematically identical to Dense (DPR) cosine similarity.
* **BM25:** 57.83% F1
* **Dense (DPR):** 60.23% F1
* **Hopfield (1-iter):** 61.23% F1

### HotpotQA (Multi-Hop Reasoning)
For multi-hop queries, we utilize `n_iter=3` to allow the network to "hop" between facts. However, early tests revealed a vulnerability: without residual connections, the iterative update pulls the query state toward whichever pattern dominates the attention distribution (attractor collapse), degrading performance. By implementing a residual update `ξ_new = normalize(ξ + retrieved)` (equivalent to a Transformer layer), stability is recovered:

* **BM25:** 14.67% F1
* **Dense (DPR):** 29.08% F1
* **Hopfield (3-iter, residual):** 26.85% F1

*Current conclusion: Continuous Hopfield Networks applied to raw text embeddings require well-separated patterns. Without a learned query-projection layer to isolate trajectories, iterative dynamics can struggle to outperform plain top-k cosine similarity.*

## 📁 Project Structure

```
hopfield_qa/
├── app.py                    # Streamlit UI
├── requirements.txt
├── benchmarks/               # SQuAD, HotpotQA, and ablation scripts
├── core/
│   ├── embeddings.py         # DocumentChunker + EmbeddingEngine
│   ├── hopfield_torch.py     # PyTorch Hopfield Memory + QA Head
│   ├── bitemporal_graph.py   # Experimental legal validity graph
│   ├── hybrid_retriever.py   # Router between Hopfield and temporal graph
│   ├── query_handler.py      # QueryProcessor (noise, expansion, variants)
│   ├── answer_generator.py   # AnswerGenerator
│   └── pipeline.py           # HopfieldQAPipeline (orchestrator)
└── utils/
    ├── visualization.py      # Plotly charts
    └── file_utils.py         # Document loading (txt, pdf, docx, md)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the App
```bash
streamlit run app.py
```

### 3. Programmatic Usage
```python
from core.pipeline import HopfieldQAPipeline, PipelineConfig

config = PipelineConfig(
    embedding_model="all-MiniLM-L6-v2",
    hopfield_beta=8.0,
    hopfield_n_iter=1,
)
pipeline = HopfieldQAPipeline(config)
```

## 🧩 Key Features

- **Modern Hopfield Retrieval** — PyTorch-backed continuous associative memory
- **Experimental Legal Router** — Route "current law" queries to a temporal graph, and vague queries to Hopfield memory (see `benchmarks/run_legal_amendments.py`)
- **Multi-Query Fusion** — Generate query variants + Reciprocal Rank Fusion
- **Pre-trained Extraction** — End-to-end extractive QA using RoBERTa
- **Rich Visualizations** — Attention heatmaps, energy landscape, PCA scatter
- **Fully offline** — No external API calls required

## 📖 References

1. Ramsauer et al. (2021). "Hopfield Networks is All You Need." ICLR 2021.
2. Hopfield, J.J. (1982). "Neural networks and physical systems with emergent collective computational abilities." PNAS.
3. Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
