# Associative Memory Retriever for Long-Context Question Answering

**Notice: This project is currently a work in progress and is under active development. Implementations, benchmarks, and architectures are subject to ongoing updates.**

An advanced question-answering system that replaces standard Retrieval-Augmented Generation (RAG) deterministic lookups with continuous Modern Hopfield Networks. Built with PyTorch, this architecture bypasses traditional Transformer token limits and mitigates conditional independence failures by utilizing energy-based recall for multi-hop reasoning.

## The Problem and Solution

Standard autoregressive language models suffer from quadratic scaling bottlenecks in their self-attention mechanisms, severely limiting their context windows. Traditional RAG systems attempt to solve this using cosine-similarity vector databases. However, because standard RAG treats every document chunk as conditionally independent, it frequently retrieves redundant information and fails at complex, multi-hop reasoning.

**Our Solution:** We implement a Dense Associative Memory using PyTorch-Hopfield layers. By treating the document corpus as a continuous energy landscape, user queries act as initial states that dynamically evolve. The query minimizes an unnormalized global energy function, gravitating toward a metastable attractor state that represents a highly coherent consensus of logically interlinked facts.

## Key Features

  * **Exponential Storage Capacity:** Leverages continuous Hopfield dynamics to store massive amounts of dense textual embeddings without catastrophic interference.
  * **Energy-Based Recall:** Replaces static Maximum Inner Product Search (MIPS) with recurrent, dynamic attractor mechanics.
  * **SQuAD Benchmarking:** Formulates massive-document QA as a Multiple Instance Learning (MIL) problem to evaluate Exact Match (EM) and F1 metrics.

## Installationbash

git clone [https://github.com/yourusername/associative-hopfield-qa.git](https://www.google.com/search?q=https://github.com/yourusername/associative-hopfield-qa.git)
cd associative-hopfield-qa
pip install -r requirements.txt

```

## Model Architecture Overview

The system relies on a decoupled architecture where the memory matrix X resides outside the generative computational graph, completely bypassing the standard LLM KV-cache limits. The update rule $\xi^{(t+1)} = X \text{softmax}(\beta X^T \xi^{(t)})$ stabilizes the query state before passing it to a lightweight extractive decoding head.
```
