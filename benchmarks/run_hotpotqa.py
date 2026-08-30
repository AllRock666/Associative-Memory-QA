import argparse
import json
import time
import torch
from tqdm import tqdm
from datasets import load_dataset
from core.embeddings import EmbeddingEngine
from core.hopfield_torch import PyTorchHopfieldMemory, BiLSTMQAHead
from core.baselines import BM25Retriever, DenseRetriever
from benchmarks.metrics import metric_max_over_ground_truths, exact_match_score, f1_score, normalize_answer

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
        scores = start_logits.unsqueeze(1) + end_logits.unsqueeze(0)
        valid_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=scores.device))
        valid_mask &= (torch.arange(L, device=scores.device).unsqueeze(1) + max_len > torch.arange(L, device=scores.device).unsqueeze(0))
        scores = scores.masked_fill(~valid_mask, -float('inf'))
        
        best = torch.argmax(scores)
        start_idx = best.item() // L
        end_idx = best.item() % L
        
    input_ids = inputs['input_ids'][0].tolist()
    
    if start_idx >= end_idx or start_idx == 0:
        return ""
        
    answer_tokens = input_ids[start_idx:end_idx+1]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Run HotpotQA benchmark on Hopfield QA")
    parser.add_argument("--subset", type=int, default=100, help="Number of samples to evaluate on (default 100)")
    parser.add_argument("--top_k", type=int, default=4, help="Top K context chunks to retrieve")
    args = parser.parse_args()

    print("Loading HotpotQA dataset (distractor setting)...")
    try:
        hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    except Exception as e:
        print(f"Error loading hotpot_qa: {e}")
        return
        
    hotpot = hotpot.select(range(args.subset))
        
    print(f"Building memory from {len(hotpot)} questions' distractors...")
    # HotpotQA context is {"title": ["t1", "t2"], "sentences": [["s1", "s2"], ["s3", "s4"]]}
    unique_paragraphs = set()
    for item in hotpot:
        titles = item['context']['title']
        sentences_lists = item['context']['sentences']
        for t, s_list in zip(titles, sentences_lists):
            para_text = "".join(s_list)
            unique_paragraphs.add(para_text)
            
    unique_contexts = list(unique_paragraphs)
    metadata = [{"chunk_id": str(i), "text": c} for i, c in enumerate(unique_contexts)]
    
    print(f"Total unique paragraphs to index: {len(unique_contexts)}")
    
    embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    patterns = []
    for c in tqdm(unique_contexts, desc="Embedding Contexts"):
        patterns.append(embedder.embed_query(c))
    import numpy as np
    patterns = np.array(patterns)
    
    print("Initializing Retrievers...")
    bm25 = BM25Retriever()
    bm25.store(unique_contexts, metadata)
    
    dpr = DenseRetriever()
    dpr.store(patterns, metadata)
    
    # Hopfield is configured with n_iter=3 for multi-hop retrieval
    hopfield = PyTorchHopfieldMemory(beta=8.0, n_iter=3)
    hopfield.store(patterns, metadata)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing QA Head (End-to-End)...")
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
    
    qa_model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2').to(device)
    qa_model.eval()

    results = {
        "BM25": {"recall": 0.0, "em": 0.0, "f1": 0.0},
        "Dense (DPR)": {"recall": 0.0, "em": 0.0, "f1": 0.0},
        "Hopfield (3-iter)": {"recall": 0.0, "em": 0.0, "f1": 0.0}
    }
    
    def evaluate_retriever(name, retriever_func, query, query_emb, ground_truths):
        if name == "BM25":
            retrieved = retriever_func(query, top_k=args.top_k)
        else:
            retrieved = retriever_func(query_emb, top_k=args.top_k)
            
        context_str = " ".join([r['text'] for r in retrieved])
        
        recall = any(normalize_answer(gt) in normalize_answer(context_str) for gt in ground_truths)
        
        pred = get_answer_span(context_str, query, qa_model, tokenizer, device)
        
        em = metric_max_over_ground_truths(exact_match_score, pred, ground_truths)
        f1 = metric_max_over_ground_truths(f1_score, pred, ground_truths)
        
        return recall, em, f1

    print("\nRunning Evaluation...")
    for item in tqdm(hotpot, desc="Evaluating Queries"):
        query = item['question']
        ground_truths = [item['answer']]
        query_emb = embedder.embed_query(query)
        
        # BM25
        r, em, f1 = evaluate_retriever("BM25", bm25.retrieve, query, query_emb, ground_truths)
        results["BM25"]["recall"] += r
        results["BM25"]["em"] += em
        results["BM25"]["f1"] += f1
        
        # Dense (DPR)
        r, em, f1 = evaluate_retriever("Dense (DPR)", dpr.retrieve, query, query_emb, ground_truths)
        results["Dense (DPR)"]["recall"] += r
        results["Dense (DPR)"]["em"] += em
        results["Dense (DPR)"]["f1"] += f1
        
        # Hopfield
        def hopfield_wrapper(q, top_k):
            res = hopfield.retrieve(q, top_k)
            return [{'text': x.text} for x in res]
            
        r, em, f1 = evaluate_retriever("Hopfield (3-iter)", hopfield_wrapper, query, query_emb, ground_truths)
        results["Hopfield (3-iter)"]["recall"] += r
        results["Hopfield (3-iter)"]["em"] += em
        results["Hopfield (3-iter)"]["f1"] += f1

    n = len(hotpot)
    for model in results:
        results[model] = {k: v / n for k, v in results[model].items()}
        
    print("\n" + "="*50)
    print("HotpotQA Benchmark Results (Multi-hop Reasoning)")
    print("="*50)
    print(f"{'Model':<20} | {'Recall@'+str(args.top_k):<10} | {'Exact Match':<11} | {'F1 Score':<10}")
    print("-"*50)
    for model, stats in results.items():
        print(f"{model:<20} | {stats['recall']*100:>8.2f}% | {stats['em']*100:>9.2f}% | {stats['f1']*100:>8.2f}%")
        
    with open("benchmarks/hotpotqa_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()
