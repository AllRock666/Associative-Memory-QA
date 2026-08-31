import json
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from core.embeddings import EmbeddingEngine
from core.hopfield_torch import PyTorchHopfieldMemory
from benchmarks.metrics import metric_max_over_ground_truths, exact_match_score, f1_score, normalize_answer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

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
    print("Loading dataset...")
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    hotpot = hotpot.select(range(50)) # 50 samples for ablation
    
    unique_paragraphs = set()
    for item in hotpot:
        titles = item['context']['title']
        sentences_lists = item['context']['sentences']
        for t, s_list in zip(titles, sentences_lists):
            para_text = "".join(s_list)
            unique_paragraphs.add(para_text)
            
    unique_contexts = list(unique_paragraphs)
    metadata = [{"chunk_id": str(i), "text": c} for i, c in enumerate(unique_contexts)]
    
    embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    patterns = []
    for c in tqdm(unique_contexts, desc="Embedding Contexts"):
        patterns.append(embedder.embed_query(c))
    patterns = np.array(patterns)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
    qa_model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2').to(device)
    qa_model.eval()

    betas = [1.0, 8.0, 50.0]
    n_iters = [1, 3]
    top_k = 4
    
    results = {}
    
    for beta in betas:
        for it in n_iters:
            name = f"Beta={beta}_Iter={it}"
            print(f"\nTesting {name}...")
            hopfield = PyTorchHopfieldMemory(beta=beta, n_iter=it, residual=True)
            hopfield.store(patterns, metadata)
            
            r_total, em_total, f1_total = 0, 0, 0
            
            for item in tqdm(hotpot, desc=f"Evaluating {name}"):
                query = item['question']
                ground_truths = [item['answer']]
                query_emb = embedder.embed_query(query)
                
                retrieved = hopfield.retrieve(query_emb, top_k=top_k)
                context_str = " ".join([r.text for r in retrieved])
                
                r = any(normalize_answer(gt) in normalize_answer(context_str) for gt in ground_truths)
                pred = get_answer_span(context_str, query, qa_model, tokenizer, device)
                em = metric_max_over_ground_truths(exact_match_score, pred, ground_truths)
                f1_score_val = metric_max_over_ground_truths(f1_score, pred, ground_truths)
                
                r_total += r
                em_total += em
                f1_total += f1_score_val
                
            n = len(hotpot)
            results[name] = {
                "recall": r_total / n,
                "em": em_total / n,
                "f1": f1_total / n
            }
            
    print("\n" + "="*50)
    print("Ablation Results")
    print("="*50)
    for name, stats in results.items():
        print(f"{name:<20} | Recall: {stats['recall']*100:.1f}% | F1: {stats['f1']*100:.1f}%")

if __name__ == "__main__":
    main()
