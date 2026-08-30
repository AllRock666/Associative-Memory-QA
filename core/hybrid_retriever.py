from typing import List, Dict, Union
from core.hopfield_torch import PyTorchHopfieldMemory
from core.bitemporal_graph import BiTemporalGraph
from core.embeddings import EmbeddingEngine

class LegalHybridRetriever:
    """
    Routes queries between a continuous associative memory (Hopfield) 
    and a discrete temporal graph based on query intent.
    """
    def __init__(self, hopfield: PyTorchHopfieldMemory, graph: BiTemporalGraph, embedder: EmbeddingEngine):
        self.hopfield = hopfield
        self.graph = graph
        self.embedder = embedder
        
        # Simple heuristics for routing. In production, this would be an LLM or fast classifier.
        self.temporal_keywords = ["current", "amended", "updated", "latest", "now", "superseded", "penalty"]

    def route_query(self, query: str) -> str:
        """Determine whether to use 'graph' or 'hopfield'."""
        q_lower = query.lower()
        if any(kw in q_lower for kw in self.temporal_keywords):
            return "graph"
        return "hopfield"

    def retrieve(self, query: str, top_k: int = 1) -> Dict:
        route = self.route_query(query)
        query_emb = self.embedder.embed_query(query)
        
        results = []
        if route == "graph":
            # Strict traversal of currently valid facts
            graph_nodes = self.graph.retrieve_current(query_emb, top_k=top_k)
            results = [{"text": n.text, "source": n.id, "route": "graph"} for n in graph_nodes]
        else:
            # Associative pattern completion over all facts
            hop_nodes = self.hopfield.retrieve(query_emb, top_k=top_k)
            results = [{"text": n.text, "source": n.chunk_id, "route": "hopfield"} for n in hop_nodes]
            
        return {
            "route": route,
            "results": results
        }
