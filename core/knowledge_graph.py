"""
Knowledge Graph Memory Engine
=============================
Implements Level 4 of the Hierarchical Knowledge Memory:
- Entity and Concept Extraction
- Relationship Triplet Mining (Subject-Predicate-Object)
- Multi-Hop Graph Traversal and Neighborhood Expansion (NetworkX)
- Cross-Document Association Mapping
- Cytoscape.js & JSON Serializers for Frontend Visualizer
"""

import re
import json
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class GraphEntity:
    """A node in the Knowledge Graph."""
    entity_id: str
    name: str
    entity_type: str = "concept"  # "concept" | "document" | "page" | "table" | "metric"
    doc_id: Optional[str] = None
    page_number: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelation:
    """A directed edge in the Knowledge Graph."""
    source_id: str
    target_id: str
    relation: str  # e.g., "defines", "depends_on", "contains", "measures", "compares_to"
    weight: float = 1.0
    provenance: Optional[str] = None


class KnowledgeGraphEngine:
    """
    Manages the multi-relational Knowledge Graph memory using NetworkX.
    Enables entity linking, sub-graph extraction, and associative neighborhood expansion.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities: Dict[str, GraphEntity] = {}
        self.relations: List[GraphRelation] = []

    def add_entity(self, entity: GraphEntity):
        self.entities[entity.entity_id] = entity
        self.graph.add_node(
            entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            doc_id=entity.doc_id,
            page_number=entity.page_number,
            **entity.attributes
        )

    def add_relation(self, relation: GraphRelation):
        self.relations.append(relation)
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation=relation.relation,
            weight=relation.weight,
            provenance=relation.provenance
        )

    def extract_and_ingest(self, chunks: List[Dict[str, Any]]):
        """
        Mines entities and relationship triplets from extracted document chunks.
        Connects document nodes, page nodes, concept nodes, and tables.
        """
        for chunk in chunks:
            text = chunk.get("text", "")
            doc_id = chunk.get("doc_id", "unknown_doc")
            doc_title = chunk.get("source", "Document")
            page_num = chunk.get("page_number", 1)
            elem_type = chunk.get("element_type", "text")

            # 1. Document Node
            doc_node_id = f"doc_{doc_id}"
            if doc_node_id not in self.entities:
                self.add_entity(GraphEntity(
                    entity_id=doc_node_id,
                    name=doc_title,
                    entity_type="document",
                    doc_id=doc_id
                ))

            # 2. Page Node
            page_node_id = f"page_{doc_id}_p{page_num}"
            if page_node_id not in self.entities:
                self.add_entity(GraphEntity(
                    entity_id=page_node_id,
                    name=f"{doc_title} (Page {page_num})",
                    entity_type="page",
                    doc_id=doc_id,
                    page_number=page_num
                ))
                self.add_relation(GraphRelation(
                    source_id=doc_node_id,
                    target_id=page_node_id,
                    relation="contains_page"
                ))

            # 3. Extract Concepts from text
            concepts = self._extract_concepts_from_text(text)
            for c in concepts:
                c_id = f"concept_{re.sub(r'[^a-zA-Z0-9_]', '', c.lower())[:32]}"
                if c_id not in self.entities:
                    self.add_entity(GraphEntity(
                        entity_id=c_id,
                        name=c,
                        entity_type="concept",
                        doc_id=doc_id,
                        page_number=page_num
                    ))
                self.add_relation(GraphRelation(
                    source_id=page_node_id,
                    target_id=c_id,
                    relation="mentions_concept",
                    weight=0.8
                ))

            # Inter-concept relationship links in same chunk
            for i in range(len(concepts) - 1):
                c1_id = f"concept_{re.sub(r'[^a-zA-Z0-9_]', '', concepts[i].lower())[:32]}"
                c2_id = f"concept_{re.sub(r'[^a-zA-Z0-9_]', '', concepts[i+1].lower())[:32]}"
                self.add_relation(GraphRelation(
                    source_id=c1_id,
                    target_id=c2_id,
                    relation="associated_in_context",
                    weight=0.6,
                    provenance=f"{doc_title} p.{page_num}"
                ))

    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Simple rule-based concept and entity extractor."""
        # Capitalized phrases and technical acronyms
        found = []
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        for a in acronyms:
            if a not in ("THE", "AND", "FOR", "NOT", "HTTP", "PDF", "URL"):
                found.append(a)

        phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        for p in phrases:
            if len(p) < 40:
                found.append(p)

        # Headings or bullet subjects
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("#") or line.startswith("Table"):
                clean = re.sub(r'^[#\*\-\s]+', '', line).strip()
                if 3 < len(clean) < 35:
                    found.append(clean)

        return list(dict.fromkeys(found))[:6]

    def expand_neighborhood(self, seed_concepts: List[str], max_hops: int = 2, max_nodes: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves the k-hop semantic neighborhood around seed entities to augment Hopfield associative recall.
        """
        matched_node_ids = set()
        for sc in seed_concepts:
            sc_clean = sc.lower()
            for nid, node_data in self.graph.nodes(data=True):
                if sc_clean in node_data.get("name", "").lower():
                    matched_node_ids.add(nid)

        if not matched_node_ids:
            return []

        expanded_nodes = set(matched_node_ids)
        for _ in range(max_hops):
            current_layer = list(expanded_nodes)
            for nid in current_layer:
                if len(expanded_nodes) >= max_nodes:
                    break
                neighbors = list(self.graph.successors(nid)) + list(self.graph.predecessors(nid))
                expanded_nodes.update(neighbors[:4])

        results = []
        for nid in expanded_nodes:
            if nid in self.entities:
                ent = self.entities[nid]
                results.append({
                    "entity_id": ent.entity_id,
                    "name": ent.name,
                    "type": ent.entity_type,
                    "page_number": ent.page_number,
                    "doc_id": ent.doc_id
                })
        return results

    def to_cytoscape(self, max_nodes: int = 60) -> Dict[str, Any]:
        """Serializes graph to Cytoscape.js format for frontend interactive exploration."""
        nodes = []
        edges = []

        subgraph_nodes = list(self.graph.nodes())[:max_nodes]
        subgraph = self.graph.subgraph(subgraph_nodes)

        color_map = {
            "document": "#6366f1",
            "page": "#3b82f6",
            "concept": "#10b981",
            "table": "#f59e0b"
        }

        for nid, data in subgraph.nodes(data=True):
            ent_type = data.get("entity_type", "concept")
            nodes.append({
                "data": {
                    "id": str(nid),
                    "label": str(data.get("name", nid)),
                    "type": ent_type,
                    "color": color_map.get(ent_type, "#8b5cf6"),
                    "page": data.get("page_number", 1)
                }
            })

        for u, v, edata in subgraph.edges(data=True):
            edges.append({
                "data": {
                    "id": f"e_{u}_{v}",
                    "source": str(u),
                    "target": str(v),
                    "label": edata.get("relation", "relates_to")
                }
            })

        return {"elements": {"nodes": nodes, "edges": edges}}
