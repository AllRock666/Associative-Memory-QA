from dataclasses import dataclass, field
import datetime
from typing import List, Dict, Optional
import numpy as np

@dataclass
class GraphNode:
    id: str
    text: str
    embedding: np.ndarray
    valid_from: datetime.date
    valid_to: Optional[datetime.date] = None
    metadata: Dict = field(default_factory=dict)
    supersedes_id: Optional[str] = None
    superseded_by_id: Optional[str] = None

class BiTemporalGraph:
    """
    An in-memory Bi-Temporal Knowledge Graph for legal documents.
    Tracks when a fact became valid and when it was superseded.
    """
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.node_counter = 0

    def add_node(self, text: str, embedding: np.ndarray, valid_from: datetime.date, metadata: Optional[Dict] = None) -> str:
        self.node_counter += 1
        node_id = f"fact_{self.node_counter}"
        self.nodes[node_id] = GraphNode(
            id=node_id,
            text=text,
            embedding=embedding,
            valid_from=valid_from,
            metadata=metadata or {}
        )
        return node_id

    def amend_node(self, old_id: str, new_text: str, new_embedding: np.ndarray, valid_from: datetime.date, metadata: Optional[Dict] = None) -> str:
        """Adds a new node that supersedes an old node."""
        if old_id not in self.nodes:
            raise ValueError(f"Node {old_id} does not exist.")
            
        old_node = self.nodes[old_id]
        old_node.valid_to = valid_from
        
        new_id = self.add_node(new_text, new_embedding, valid_from, metadata)
        new_node = self.nodes[new_id]
        
        old_node.superseded_by_id = new_id
        new_node.supersedes_id = old_id
        
        return new_id

    def retrieve_current(self, query_embedding: np.ndarray, top_k: int = 1, as_of_date: datetime.date = None) -> List[GraphNode]:
        """
        Traverses the graph to find nodes valid as of the given date (default: today),
        then ranks them by similarity to the query.
        """
        if as_of_date is None:
            as_of_date = datetime.date.today()
            
        valid_nodes = []
        for node in self.nodes.values():
            if node.valid_from <= as_of_date:
                if node.valid_to is None or node.valid_to > as_of_date:
                    valid_nodes.append(node)
                    
        if not valid_nodes:
            return []
            
        # Standard cosine similarity over the VALID subset
        patterns = np.array([n.embedding for n in valid_nodes])
        norms = np.linalg.norm(patterns, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        patterns = patterns / norms
        
        q = query_embedding.copy()
        n = np.linalg.norm(q)
        if n > 1e-8:
            q = q / n
            
        similarities = patterns @ q
        
        top_k = min(top_k, len(valid_nodes))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [valid_nodes[idx] for idx in top_indices]
