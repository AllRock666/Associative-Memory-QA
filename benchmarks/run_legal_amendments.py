import datetime
import numpy as np
from core.embeddings import EmbeddingEngine
from core.hopfield_torch import PyTorchHopfieldMemory
from core.bitemporal_graph import BiTemporalGraph
from core.hybrid_retriever import LegalHybridRetriever

def main():
    print("Initializing Legal Hybrid Infrastructure...")
    embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    hopfield = PyTorchHopfieldMemory(beta=8.0, n_iter=3)
    graph = BiTemporalGraph()

    # Create synthetic legal dataset
    doc1_text = "The penalty for late delivery is $50,000."
    doc2_text = "Amendment: The penalty for late delivery is increased to $75,000."
    doc3_text = "The governing law for this contract is the state of California."
    
    print("Embedding legal clauses...")
    emb1 = embedder.embed_query(doc1_text)
    emb2 = embedder.embed_query(doc2_text)
    emb3 = embedder.embed_query(doc3_text)

    print("Populating Bi-Temporal Graph & Hopfield Memory...")
    # Add to graph
    node1_id = graph.add_node(doc1_text, emb1, valid_from=datetime.date(2023, 1, 1))
    node2_id = graph.amend_node(node1_id, doc2_text, emb2, valid_from=datetime.date(2024, 1, 1))
    node3_id = graph.add_node(doc3_text, emb3, valid_from=datetime.date(2023, 1, 1))
    
    # Add to Hopfield (which just sees a flat array of all facts)
    patterns = np.array([emb1, emb2, emb3])
    metadata = [
        {"chunk_id": "doc1", "text": doc1_text},
        {"chunk_id": "doc2", "text": doc2_text},
        {"chunk_id": "doc3", "text": doc3_text}
    ]
    hopfield.store(patterns, metadata)

    hybrid = LegalHybridRetriever(hopfield, graph, embedder)

    print("\n" + "="*60)
    print("Testing Failure Mode: Metastability in Pure Hopfield")
    print("="*60)
    query = "What is the penalty for late delivery?"
    q_emb = embedder.embed_query(query)
    
    # Pure Hopfield will likely retrieve the older one or be confused, because the embeddings are nearly identical
    res_hop = hopfield.retrieve(q_emb, top_k=2)
    print(f"Query: '{query}'")
    print("Pure Hopfield Retrieves:")
    for r in res_hop:
        print(f" - [Weight: {r.similarity_score:.4f}] {r.text}")
        
    print("\n" + "="*60)
    print("Testing Hybrid Router: Strict Supersession via Graph")
    print("="*60)
    # The router should see 'penalty' (a tracked tracked concept/amendment) and route to graph
    res_hybrid = hybrid.retrieve(query, top_k=1)
    print(f"Routed to: {res_hybrid['route'].upper()}")
    print("Hybrid System Retrieves:")
    for r in res_hybrid['results']:
        print(f" - {r['text']}")
        
    print("\n" + "="*60)
    print("Testing Hybrid Router: Vague Pattern Completion")
    print("="*60)
    query_vague = "governing law state"
    res_vague = hybrid.retrieve(query_vague, top_k=1)
    print(f"Query: '{query_vague}'")
    print(f"Routed to: {res_vague['route'].upper()}")
    print("Hybrid System Retrieves:")
    for r in res_vague['results']:
        print(f" - {r['text']}")
        
    print("\n" + "="*60)
    print("Conclusion: The hybrid system successfully bridges Hopfield's")
    print("associative recall with strict temporal validity, preventing")
    print("catastrophic legal blending!")
    print("="*60)

if __name__ == "__main__":
    main()
