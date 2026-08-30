import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from core.embeddings import EmbeddingEngine
from core.hopfield_torch import PyTorchHopfieldMemory
from core.baselines import DenseRetriever

def normalize_text(t):
    import re
    return re.sub(r'\s+', ' ', t).strip().lower()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=100)
    args = parser.parse_args()

    print("Loading HotpotQA (validation)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    dataset = dataset.select(range(args.subset))
    
    embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    
    dpr = DenseRetriever()
    # Let's test standard DPR vs Hopfield (n_iter=3)
    hopfield = PyTorchHopfieldMemory(beta=8.0, n_iter=3)
    
    results = {
        "Dense (DPR)": {"strict_success": 0.0, "partial_success": 0.0},
        "Hopfield (n_iter=3)": {"strict_success": 0.0, "partial_success": 0.0}
    }
    
    print("Evaluating Multi-hop Retrieval...")
    for item in tqdm(dataset):
        query = item['question']
        query_emb = embedder.embed_query(query)
        
        # Supporting facts are titles of the correct paragraphs
        supporting_titles = set(item['supporting_facts']['title'])
        
        # The context in HotpotQA is a list of paragraphs (each has a title and list of sentences)
        contexts = []
        for title, sentences in zip(item['context']['title'], item['context']['sentences']):
            text = " ".join(sentences)
            contexts.append({"chunk_id": title, "text": text, "source": title})
            
        patterns = np.array([embedder.embed_query(c["text"]) for c in contexts])
        
        dpr.store(patterns, contexts)
        hopfield.store(patterns, contexts)
        
        # Retrieve top-k = 2 (since usually 2 supporting facts)
        top_k = 2
        dpr_retrieved = dpr.retrieve(query_emb, top_k=top_k)
        hop_retrieved = hopfield.retrieve(query_emb, top_k=top_k)
        
        dpr_titles = set(r['chunk_id'] for r in dpr_retrieved)
        hop_titles = set(r.chunk_id for r in hop_retrieved)
        
        # Strict success: both supporting facts retrieved
        # Partial success: at least one supporting fact retrieved
        
        if supporting_titles.issubset(dpr_titles):
            results["Dense (DPR)"]["strict_success"] += 1
        if len(supporting_titles.intersection(dpr_titles)) > 0:
            results["Dense (DPR)"]["partial_success"] += 1
            
        if supporting_titles.issubset(hop_titles):
            results["Hopfield (n_iter=3)"]["strict_success"] += 1
        if len(supporting_titles.intersection(hop_titles)) > 0:
            results["Hopfield (n_iter=3)"]["partial_success"] += 1

    n = len(dataset)
    for model in results:
        results[model] = {k: v / n for k, v in results[model].items()}
        
    print("\n" + "="*50)
    print("HotpotQA Multi-Hop Retrieval Benchmark (Top-2)")
    print("="*50)
    print(f"{'Model':<20} | {'Partial (>=1)':<15} | {'Strict (Both)':<15}")
    print("-"*50)
    for model, stats in results.items():
        print(f"{model:<20} | {stats['partial_success']*100:>14.2f}% | {stats['strict_success']*100:>14.2f}%")
        
if __name__ == "__main__":
    main()
