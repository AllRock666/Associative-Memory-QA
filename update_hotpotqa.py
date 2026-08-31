import re

def update_hotpotqa():
    with open('benchmarks/run_hotpotqa.py', 'r') as f:
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

    # Add titles to metadata
    old_meta = """    unique_contexts = list(unique_paragraphs)
    metadata = [{"chunk_id": str(i), "text": c} for i, c in enumerate(unique_contexts)]"""
    new_meta = """    unique_paragraphs_dict = {}
    for item in hotpot:
        titles = item['context']['title']
        sentences_lists = item['context']['sentences']
        for t, s_list in zip(titles, sentences_lists):
            para_text = "".join(s_list)
            unique_paragraphs_dict[para_text] = t
            
    unique_contexts = list(unique_paragraphs_dict.keys())
    metadata = [{"chunk_id": str(i), "title": unique_paragraphs_dict[c], "text": c} for i, c in enumerate(unique_contexts)]"""
    
    # We replace the set building and metadata
    old_para = """    unique_paragraphs = set()
    for item in hotpot:
        titles = item['context']['title']
        sentences_lists = item['context']['sentences']
        for t, s_list in zip(titles, sentences_lists):
            para_text = "".join(s_list)
            unique_paragraphs.add(para_text)
            
    unique_contexts = list(unique_paragraphs)
    metadata = [{"chunk_id": str(i), "text": c} for i, c in enumerate(unique_contexts)]"""
    content = content.replace(old_para, new_meta)

    # Update evaluating
    old_eval = """    def evaluate_retriever(name, retriever_func, query, query_emb, ground_truths):
        if name == "BM25":
            retrieved = retriever_func(query, top_k=args.top_k)
        else:
            retrieved = retriever_func(query_emb, top_k=args.top_k)
            
        context_str = " ".join([r['text'] for r in retrieved])
        
        recall = any(normalize_answer(gt) in normalize_answer(context_str) for gt in ground_truths)
        
        pred = get_answer_span(context_str, query, qa_model, tokenizer, device)
        
        em = metric_max_over_ground_truths(exact_match_score, pred, ground_truths)
        f1 = metric_max_over_ground_truths(f1_score, pred, ground_truths)
        
        return recall, em, f1"""

    new_eval = """    def evaluate_retriever(name, retriever_func, query, query_emb, ground_truths, support_titles):
        if name == "BM25":
            retrieved = retriever_func(query, top_k=args.top_k)
        else:
            retrieved = retriever_func(query_emb, top_k=args.top_k)
            
        context_str = truncate_chunks_for_roberta(retrieved, query, tokenizer)
        
        retrieved_titles = [r.get('title', '') for r in retrieved]
        
        recall = 1.0 if any(t in retrieved_titles for t in support_titles) else 0.0
        
        mrr = 0.0
        for i, t in enumerate(retrieved_titles):
            if t in support_titles:
                mrr = 1.0 / (i + 1)
                break
        
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
        def hopfield_wrapper(q, top_k):
            res = hopfield.retrieve(q, top_k)
            return [{'text': x.text} for x in res]
            
        r, em, f1 = evaluate_retriever("Hopfield (3-iter)", hopfield_wrapper, query, query_emb, ground_truths)
        results["Hopfield (3-iter)"]["recall"] += r
        results["Hopfield (3-iter)"]["em"] += em
        results["Hopfield (3-iter)"]["f1"] += f1"""

    new_call = """        support_titles = item['supporting_facts']['title']
        
        # BM25
        r, mrr, em, f1 = evaluate_retriever("BM25", bm25.retrieve, query, query_emb, ground_truths, support_titles)
        results["BM25"]["recall"] += r
        results["BM25"]["mrr"] += mrr
        results["BM25"]["em"] += em
        results["BM25"]["f1"] += f1
        
        # Dense (DPR)
        r, mrr, em, f1 = evaluate_retriever("Dense (DPR)", dpr.retrieve, query, query_emb, ground_truths, support_titles)
        results["Dense (DPR)"]["recall"] += r
        results["Dense (DPR)"]["mrr"] += mrr
        results["Dense (DPR)"]["em"] += em
        results["Dense (DPR)"]["f1"] += f1
        
        # Hopfield
        def hopfield_wrapper(q, top_k):
            res = hopfield.retrieve(q, top_k)
            return [{'title': x.source, 'text': x.text} for x in res]
            
        r, mrr, em, f1 = evaluate_retriever("Hopfield (3-iter)", hopfield_wrapper, query, query_emb, ground_truths, support_titles)
        results["Hopfield (3-iter)"]["recall"] += r
        results["Hopfield (3-iter)"]["mrr"] += mrr
        results["Hopfield (3-iter)"]["em"] += em
        results["Hopfield (3-iter)"]["f1"] += f1"""

    content = content.replace(old_call, new_call)

    # Oh, wait! The hopfield_wrapper needs title.
    # PyTorchHopfieldMemory.store creates chunks.
    # In run_hotpotqa.py: hopfield.store(patterns, metadata)
    # The chunk.source is set to chunk.metadata.get('source', '') or 'unknown'
    # BUT in metadata above we used "title". So chunk.metadata.get("title") is what we need.
    content = content.replace("'title': x.source", "'title': x.metadata.get('title', '')")

    # The title for baselines: BM25/Dense return dicts directly from the metadata list!
    # Yes, bm25.retrieve returns dicts with 'text', 'title', etc.

    content = re.sub(r'print\(f"\{\'Model\':<20\}.*?\)', 'print(f"{\'Model\':<20} | {\'Recall@\'+str(args.top_k):<10} | {\'MRR\':<10} | {\'Exact Match\':<11} | {\'F1 Score\':<10}")', content)
    content = re.sub(r'print\(f"\{model:<20\}.*?\)', 'print(f"{model:<20} | {stats[\'recall\']*100:>8.2f}% | {stats[\'mrr\']:>10.4f} | {stats[\'em\']*100:>9.2f}% | {stats[\'f1\']*100:>8.2f}%")', content)
    content = content.replace('print("="*50)', 'print("="*70)')
    content = content.replace('print("-"'*50 + ')', 'print("-"'*70 + ')')
    content = content.replace('print("-" * 50)', 'print("-" * 70)')
    content = content.replace('print("=" * 50)', 'print("=" * 70)')
    content = content.replace('print("-"*50)', 'print("-"*70)')

    with open('benchmarks/run_hotpotqa.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    update_hotpotqa()
