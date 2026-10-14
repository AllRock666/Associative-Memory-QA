import argparse
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import mlflow
from core.embeddings import EmbeddingEngine
from core.hopfield_torch import PyTorchHopfieldMemory, BiLSTMQAHead
from core.baselines import BM25Retriever, DenseRetriever
from benchmarks.metrics import metric_max_over_ground_truths, exact_match_score, f1_score, normalize_answer

def truncate_chunks_for_roberta(chunks, query, tokenizer, max_total_tokens=512):
    query_tokens = len(tokenizer(query, add_special_tokens=False)['input_ids'])
    special_tokens = 4
    budget = max_total_tokens - query_tokens - special_tokens
    if budget <= 0: budget = 10
    budget_per_chunk = budget // max(len(chunks), 1)
    
    truncated_texts = []
    for chunk in chunks:
        ids = tokenizer((chunk.text if hasattr(chunk, 'text') else (chunk.text if hasattr(chunk, 'text') else chunk['text'])), add_special_tokens=False)['input_ids']
        if len(ids) > budget_per_chunk:
            ids = ids[:budget_per_chunk]
        truncated_texts.append(tokenizer.decode(ids))
    return " ".join(truncated_texts)

def get_answer_span(context, query, qa_model, tokenizer, device):
    inputs = tokenizer(query, context, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = qa_model(**inputs)
        
        start_logits = outputs.start_logits.squeeze(0)
        end_logits = outputs.end_logits.squeeze(0)
        
        sequence_ids = inputs.get('token_type_ids', None)
        if sequence_ids is None:
            query_len = len(tokenizer.tokenize(query)) + 2
            mask = torch.ones_like(start_logits, dtype=torch.bool)
            mask[:query_len] = False
        else:
            mask = (sequence_ids.squeeze(0) == 1)
            
        start_logits = start_logits.masked_fill(~mask, -float('inf'))
        end_logits = end_logits.masked_fill(~mask, -float('inf'))
        
        L = start_logits.size(0)
        max_len = 30
        
        score_matrix = start_logits.unsqueeze(1) + end_logits.unsqueeze(0)
        
        mask_upper = torch.triu(torch.ones(L, L, device=device), diagonal=0).bool()
        mask_len = torch.tril(torch.ones(L, L, device=device), diagonal=max_len).bool()
        valid_mask = mask_upper & mask_len
        
        score_matrix = score_matrix.masked_fill(~valid_mask, -float('inf'))
        
        flat_idx = torch.argmax(score_matrix)
        start_idx = flat_idx // L
        end_idx = flat_idx % L
        
    answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
    return tokenizer.decode(answer_tokens)

def evaluate_retriever(retriever_name, retriever_obj, data, qa_model, tokenizer, device, top_k=3, embedder=None, args=None):
    results = {"em": 0.0, "f1": 0.0, "time_ms": 0.0, "recall": 0.0, "mrr": 0.0}
    total = len(data)
    start = time.time()
    
    with mlflow.start_run(run_name=f"SQuAD-{retriever_name}", nested=True):
        mlflow.log_params({
            "retriever": retriever_name,
            "top_k": top_k,
            "dataset": "SQuAD_v2",
            "subset": args.subset if args else None
        })
        if args and retriever_name == "Hopfield":
            mlflow.log_params({
                "hopfield_beta": args.beta,
                "hopfield_n_iter": args.n_iter,
                "energy_type": getattr(args, "energy_type", "softmax"),
                "hopfield_use_projection": getattr(args, "use_projection", False)
            })

        for item in tqdm(data, desc=f"Evaluating {retriever_name}"):
            query = item['question']
            answers = item['answers']['text'] if isinstance(item['answers'], dict) else item['answers']
            if not answers:
                total -= 1
                continue
                
            if embedder:
                q_emb = embedder.embed_query(query)
                chunks = retriever_obj.retrieve(q_emb, top_k=top_k)
            else:
                chunks = retriever_obj.retrieve(query, top_k=top_k)
                
            retrieved_texts = [(c.text if hasattr(c, 'text') else (c.text if hasattr(c, 'text') else c['text'])) for c in chunks]
            
            recall_found = False
            for text in retrieved_texts:
                if any(normalize_answer(ans) in normalize_answer(text) for ans in answers):
                    recall_found = True
                    break
            if recall_found:
                results["recall"] += 1.0
                
            mrr_found = False
            for rank, text in enumerate(retrieved_texts):
                if any(normalize_answer(ans) in normalize_answer(text) for ans in answers):
                    results["mrr"] += 1.0 / (rank + 1)
                    mrr_found = True
                    break
            
            context = truncate_chunks_for_roberta(chunks, query, tokenizer)
            pred = get_answer_span(context, query, qa_model, tokenizer, device)
            
            results["em"] += metric_max_over_ground_truths(exact_match_score, pred, answers)
            results["f1"] += metric_max_over_ground_truths(f1_score, pred, answers)
            
        elapsed = time.time() - start
        if total > 0:
            results["em"] /= total
            results["f1"] /= total
            results["recall"] /= total
            results["mrr"] /= total
            results["time_ms"] = (elapsed * 1000) / total
            
        mlflow.log_metrics(results)
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--energy_type", type=str, default="softmax")
    parser.add_argument("--use_projection", action="store_true")
    parser.add_argument("--projection_path", type=str, default="models/projection_squad.pt")
    args = parser.parse_args()

    mlflow.set_experiment("HopfieldQA-SQuAD-Benchmark")

    dataset = load_dataset("squad_v2", split="validation")
    dataset = dataset.shuffle(seed=args.seed).select(range(args.subset))
    
    embedder = EmbeddingEngine()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    
    print("Building memory...")
    contexts = list(set([item['context'] for item in dataset]))
    
    chunks = []
    patterns = []
    metadata = []
    
    for i, ctx in enumerate(tqdm(contexts)):
        emb = embedder.embed_query(ctx)
        patterns.append(emb)
        metadata.append({"chunk_id": str(i), "text": ctx})
        chunks.append({"id": str(i), "text": ctx})
        
    patterns_tensor = np.array(patterns)
    
    bm25 = BM25Retriever()
    bm25.store([(c.text if hasattr(c, 'text') else (c.text if hasattr(c, 'text') else c['text'])) for c in chunks], metadata)
    
    dense = DenseRetriever()
    dense.store(patterns_tensor, metadata)
    
    hopfield = PyTorchHopfieldMemory(beta=args.beta, n_iter=args.n_iter, use_projection=args.use_projection)
    if args.use_projection and args.projection_path:
        import os
        if os.path.exists(args.projection_path):
            hopfield.retriever.projection.load_state_dict(torch.load(args.projection_path, map_location=hopfield.device, weights_only=True))
            print(f"Loaded projection from {args.projection_path}")
        else:
            print(f"Warning: Projection path {args.projection_path} not found.")
    hopfield.retriever.energy_type = args.energy_type
    hopfield.store(patterns_tensor, metadata)
    
    results = {}
    with mlflow.start_run(run_name=f"Benchmark-Suite-{args.subset}"):
        mlflow.log_params(vars(args))
        
        results["BM25"] = evaluate_retriever("BM25", bm25, dataset, qa_model, tokenizer, device, top_k=args.top_k, args=args)
        results["Dense"] = evaluate_retriever("Dense", dense, dataset, qa_model, tokenizer, device, top_k=args.top_k, embedder=embedder, args=args)
        results["Hopfield"] = evaluate_retriever("Hopfield", hopfield, dataset, qa_model, tokenizer, device, top_k=args.top_k, embedder=embedder, args=args)
    
    with open("benchmarks/squad_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()



