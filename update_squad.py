import re

def update_squad():
    with open('benchmarks/run_squad.py', 'r') as f:
        content = f.read()

    truncate_func = """
def truncate_chunks_for_roberta(chunks, query, tokenizer, max_total_tokens=512):
    query_tokens = len(tokenizer(query, add_special_tokens=False)['input_ids'])
    special_tokens = 4
    budget = max_total_tokens - query_tokens - special_tokens
    if budget <= 0: budget = 10
    budget_per_chunk = budget // max(len(chunks), 1)
    
    truncated_texts = []
    for chunk in chunks:
        ids = tokenizer(chunk['text'], add_special_tokens=False)['input_ids']
        if len(ids) > budget_per_chunk:
            ids = ids[:budget_per_chunk]
        truncated_texts.append(tokenizer.decode(ids))
    return " ".join(truncated_texts)
"""
    if "def truncate_chunks_for_roberta" not in content:
        content = content.replace("def get_answer_span(", truncate_func + "\ndef get_answer_span(")

    old_eval = """    def evaluate_retriever(name, retriever_func, query, query_emb, ground_truths):
        if name == "BM25":
            retrieved = retriever_func(query, top_k=args.top_k)
        else:
            retrieved = retriever_func(query_emb, top_k=args.top_k)
            
        context_str = " ".join([r['text'] for r in retrieved])
        
        # Retrieval Recall: is any valid answer string present in the retrieved context?
        recall = any(normalize_answer(gt) in normalize_answer(context_str) for gt in ground_truths)
        
        # End-to-End prediction
        pred = get_answer_span(context_str, query, qa_model, tokenizer, device)
        
        em = metric_max_over_ground_truths(exact_match_score, pred, ground_truths)
        f1 = metric_max_over_ground_truths(f1_score, pred, ground_truths)
        
        return recall, em, f1"""

    new_eval = """    def evaluate_retriever(name, retriever_func, query, query_emb, ground_truths):
        if name == "BM25":
            retrieved = retriever_func(query, top_k=args.top_k)
        else:
            retrieved = retriever_func(query_emb, top_k=args.top_k)
            
        context_str = truncate_chunks_for_roberta(retrieved, query, tokenizer)
        
        # Pure Retrieval Metric: string matching on chunks
        recall = 0.0
        mrr = 0.0
        for i, r in enumerate(retrieved):
            if any(normalize_answer(gt) in normalize_answer(r['text']) for gt in ground_truths):
                recall = 1.0
                mrr = 1.0 / (i + 1)
                break
        
        # End-to-End prediction
        pred = get_answer_span(context_str, query, qa_model, tokenizer, device)
        
        em = metric_max_over_ground_truths(exact_match_score, pred, ground_truths)
        f1 = metric_max_over_ground_truths(f1_score, pred, ground_truths)
        
        return recall, mrr, em, f1"""
    
    content = content.replace(old_eval, new_eval)

    content = content.replace('"recall": 0.0, "em"', '"recall": 0.0, "mrr": 0.0, "em"')

    old_call = """        # BM25
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
        # Adapter retrieve returns objects, make it dict compatible
        def hopfield_wrapper(q, top_k):
            res = hopfield.retrieve(q, top_k)
            return [{'text': x.text} for x in res]
            
        r, em, f1 = evaluate_retriever("Hopfield", hopfield_wrapper, query, query_emb, ground_truths)
        results["Hopfield"]["recall"] += r
        results["Hopfield"]["em"] += em
        results["Hopfield"]["f1"] += f1"""

    new_call = """        # BM25
        r, mrr, em, f1 = evaluate_retriever("BM25", bm25.retrieve, query, query_emb, ground_truths)
        results["BM25"]["recall"] += r
        results["BM25"]["mrr"] += mrr
        results["BM25"]["em"] += em
        results["BM25"]["f1"] += f1
        
        # Dense (DPR)
        r, mrr, em, f1 = evaluate_retriever("Dense (DPR)", dpr.retrieve, query, query_emb, ground_truths)
        results["Dense (DPR)"]["recall"] += r
        results["Dense (DPR)"]["mrr"] += mrr
        results["Dense (DPR)"]["em"] += em
        results["Dense (DPR)"]["f1"] += f1
        
        # Hopfield
        # Adapter retrieve returns objects, make it dict compatible
        def hopfield_wrapper(q, top_k):
            res = hopfield.retrieve(q, top_k)
            return [{'text': x.text} for x in res]
            
        r, mrr, em, f1 = evaluate_retriever("Hopfield", hopfield_wrapper, query, query_emb, ground_truths)
        results["Hopfield"]["recall"] += r
        results["Hopfield"]["mrr"] += mrr
        results["Hopfield"]["em"] += em
        results["Hopfield"]["f1"] += f1"""

    content = content.replace(old_call, new_call)

    content = re.sub(r'print\(f"\{\'Model\':<15\}.*?\)', 'print(f"{\'Model\':<15} | {\'Recall@\'+str(args.top_k):<10} | {\'MRR\':<10} | {\'Exact Match\':<11} | {\'F1 Score\':<10}")', content)
    content = re.sub(r'print\(f"\{model:<15\}.*?\)', 'print(f"{model:<15} | {stats[\'recall\']*100:>8.2f}% | {stats[\'mrr\']:>10.4f} | {stats[\'em\']*100:>9.2f}% | {stats[\'f1\']*100:>8.2f}%")', content)
    content = content.replace('print("="*50)', 'print("="*65)')
    content = content.replace('print("-"'*50 + ')', 'print("-"'*65 + ')')
    content = content.replace('print("-" * 50)', 'print("-" * 65)')
    content = content.replace('print("=" * 50)', 'print("=" * 65)')
    content = content.replace('print("-"*50)', 'print("-"*65)')

    with open('benchmarks/run_squad.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    update_squad()
