import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from core.embeddings import EmbeddingEngine
from core.hopfield_torch import PyTorchHopfieldMemory
from core.baselines import DenseRetriever
from benchmarks.metrics import normalize_answer

def inject_noise(query, noise_level=0.2, seed=42):
    random.seed(seed)
    words = query.split()
    if not words: return query
    
    # Simple typo injection
    chars = "abcdefghijklmnopqrstuvwxyz"
    num_to_corrupt = int(max(1, len(words) * noise_level))
    indices = random.sample(range(len(words)), min(num_to_corrupt, len(words)))
    
    for idx in indices:
        word = words[idx]
        if len(word) > 3:
            char_idx = random.randint(1, len(word) - 2)
            word = word[:char_idx] + random.choice(chars) + word[char_idx+1:]
        words[idx] = word
        
    # Truncation at end
    if noise_level > 0.3 and len(words) > 3:
        words = words[:-1]
        
    return " ".join(words)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hopfield vs Dense on Noisy Queries")
    parser.add_argument("--subset", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print("Loading SQuAD dataset...")
    squad = load_dataset("squad", split="validation")
    squad = squad.select(range(args.subset))
        
    unique_contexts = list(set(squad["context"]))
    metadata = [{"chunk_id": str(i), "text": c} for i, c in enumerate(unique_contexts)]
    
    embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    patterns = []
    for c in tqdm(unique_contexts, desc="Embedding Contexts"):
        patterns.append(embedder.embed_query(c))
    patterns = np.array(patterns)
    
    print("Initializing Retrievers...")
    dpr = DenseRetriever()
    dpr.store(patterns, metadata)
    
    hopfield = PyTorchHopfieldMemory(beta=8.0, n_iter=3, similarity_threshold=0.0)
    hopfield.store(patterns, metadata)
    
    # We will use Multi-Query Retrieval from QueryProcessor just to see fusion in action
    from core.query_handler import QueryProcessor
    processor = QueryProcessor()

    def evaluate_retriever(retriever_func, query_variants, ground_truths):
        # We will embed the variants, retrieve, and fuse (simplified RRF)
        all_retrieved = []
        for v in query_variants:
            emb = embedder.embed_query(v)
            retrieved = retriever_func(emb, top_k=args.top_k)
            all_retrieved.append(retrieved)
            
        # Simplified fusion: just take the top-k from the first variant (since they are list of dicts/objects)
        # Proper fusion requires ID tracking, but let's keep it simple and just pool and rank by original score
        pooled = []
        for r_list in all_retrieved:
            for r in r_list:
                if type(r) is dict:
                    pooled.append((r.get('similarity_score', 0.0), r['text']))
                else:
                    pooled.append((r.similarity_score, r.text))
                    
        pooled.sort(key=lambda x: x[0], reverse=True)
        # Deduplicate
        seen = set()
        final_retrieved = []
        for _, text in pooled:
            if text not in seen:
                seen.add(text)
                final_retrieved.append(text)
            if len(final_retrieved) >= args.top_k: break
            
        recall = 1.0 if any(any(normalize_answer(gt) in normalize_answer(t) for gt in ground_truths) for t in final_retrieved) else 0.0
        return recall

    print("\nRunning Noisy Evaluation...")
    
    results = {nl: {"Dense": 0.0, "Hopfield": 0.0} for nl in args.noise_levels}
    total = len(squad) * args.repeats
    
    for item in tqdm(squad, desc="Evaluating Queries"):
        orig_query = item['question']
        ground_truths = item['answers']['text']
        
        for nl in args.noise_levels:
            for rep in range(args.repeats):
                seed = hash(orig_query) + rep
                noisy_query = inject_noise(orig_query, nl, seed=seed)
                
                # Variants via processor
                pq = processor.process(noisy_query)
                variants = pq.variants
                
                # Dense
                r_dense = evaluate_retriever(dpr.retrieve, variants, ground_truths)
                results[nl]["Dense"] += r_dense
                
                # Hopfield
                r_hop = evaluate_retriever(hopfield.retrieve, variants, ground_truths)
                results[nl]["Hopfield"] += r_hop

    print("\n" + "="*50)
    print("Noisy Query Robustness (Recall@3)")
    print("="*50)
    print(f"{'Noise Level':<15} | {'Dense (Fusion)':<15} | {'Hopfield (Fusion)':<15}")
    print("-" * 50)
    for nl in args.noise_levels:
        d_rec = results[nl]["Dense"] / total
        h_rec = results[nl]["Hopfield"] / total
        print(f"{nl:<15} | {d_rec*100:>14.2f}% | {h_rec*100:>16.2f}%")
        
if __name__ == "__main__":
    main()
