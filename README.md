# 🧠 Associative Memory Retriever for Long-Context QA
### Exploring Modern Hopfield Networks for Retrieval

**The HAM Engine transitions document retrieval from isolated vector similarity lookup into continuous associative pattern completion over energy landscapes.**

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
With a trained query projection layer (`use_projection=True`), Hopfield matches or exceeds DPR.

*(Run with `python benchmarks/run_squad.py --subset 100 --use_projection`, seed=42)*
* **BM25:** 87.66% F1 (Recall@3: 95.83%, MRR: 0.920)
* **Dense (DPR):** 87.95% F1 (Recall@3: 100%, MRR: 0.990)
* **Hopfield (Learned Projection):** 87.65% F1 (Recall@3: 100%, MRR: 0.938)

### HotpotQA (Multi-Hop Reasoning)
For multi-hop queries, we utilize `n_iter=3` to allow the network to "hop" between facts. However, early tests revealed a vulnerability: without residual connections, the iterative update pulls the query state toward whichever pattern dominates the attention distribution (attractor collapse), degrading performance. By implementing a residual update `x_new = normalize(x + retrieved)` (equivalent to a Transformer layer), stability is recovered, and with our learned projection layer, pure recall exceeds Dense retrieval:

*(Run with `python benchmarks/run_hotpotqa.py --subset 100 --use_projection`, seed=42)*
* **BM25:** 32.38% F1 (Recall: 91%, MRR: 0.80)
* **Dense (DPR):** 32.60% F1 (Recall: 94%, MRR: 0.87)
* **Hopfield (3-iter, Learned Projection):** 32.02% F1 (Recall: 96%, MRR: 0.81)

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
## Docker Deployment

You can easily package and run the entire Hopfield QA application using Docker. This ensures that all dependencies (like PyPDF and PyTorch) and models are correctly bundled.

### 1. Build the Docker Image
This process downloads the embedding model used by the API into the image, avoiding a model download at startup.
```bash
docker build -t hopfield-qa .
```

### 2. Run the Container
Run the container and map the internal port (8080) to your machine's port:
```bash
docker run -p 8080:8080 hopfield-qa
```

Once running, simply open your browser and go to:
**http://localhost:8080**

## Free-tier deployment on Google Cloud Run

Cloud Run has a monthly free allowance for request-based services, which can cover a small demo; the allowance and eligible regions are described in the [Cloud Run pricing docs](https://cloud.google.com/run/pricing). A Google Cloud billing account is still required, and usage beyond the free allowance can be charged. Source deployments also use Cloud Build and Artifact Registry, which have separate pricing and free allowances. Set a budget alert in Google Cloud before deploying; budget alerts notify you but do not cap spending.

1. Install the Google Cloud CLI, sign in with `gcloud auth login`, create/select a project, and enable billing for that project.
2. From the repository root in PowerShell, select your project and deploy:

   ```powershell
   gcloud config set project YOUR_PROJECT_ID
   gcloud run deploy hopfield-qa `
     --source . `
     --region asia-south1 `
     --allow-unauthenticated `
     --memory 2Gi `
     --cpu 2 `
     --concurrency 1 `
     --min 0 `
     --max 1 `
     --timeout 300 `
     --set-env-vars LLM_PROVIDER=mock,LLM_FALLBACK_TO_DETERMINISTIC=true,WEB_SEARCH_ENABLED=false
   ```

3. Accept the one-time API enablement prompts. When deployment finishes, open the HTTPS service URL printed by `gcloud`.

The container listens on Cloud Run's injected `PORT` as required by the [container runtime contract](https://cloud.google.com/run/docs/container-contract) and uses the built-in grounded mock provider, so no Ollama server or API key is needed. Cloud Run instances can scale to zero and have an ephemeral filesystem: uploaded documents, conversations, and learned index state are lost when an instance stops. This application also keeps one shared in-memory document index and chat store, with no authentication or per-user isolation. Use it only for public demo documents until persistent storage and user isolation are implemented. Public Cloud Run access means anyone with the URL can use the app and its API.

