"""
Evidence Deduplication Subsystem
================================
Removes exact and near-duplicate evidence passages before selection and synthesis:
1. Normalizes whitespace, casing, and punctuation for robust comparison.
2. Uses exact hashing and token / n-gram Jaccard similarity for near-duplicate clustering.
3. Preserves the highest-scoring representative chunk and merges provenance metadata
   (source documents, chunk IDs, page numbers) so attribution is never lost.
4. Distinguishes near-duplicates from distinct passages discussing the same topic.
5. Keeps the original retrieval ranking and Hopfield trajectory diagnostics observable.
"""

import re
import copy
import logging
from typing import List, Dict, Any, Tuple, Set, Optional
from core.schemas import EvidenceChunk, EvidenceCollection

logger = logging.getLogger(__name__)


class EvidenceDeduplicator:
    """
    Identifies and collapses duplicate and near-duplicate evidence chunks while
    preserving source provenance and retrieval diagnostics.
    """

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalizes text by removing punctuation, lowercasing, and collapsing whitespace."""
        clean = re.sub(r'[^\w\s]', ' ', text.lower())
        return ' '.join(clean.split())

    @staticmethod
    def get_word_tokens(text: str) -> List[str]:
        """Extracts word tokens from text."""
        return [w for w in re.findall(r'\b\w+\b', text.lower()) if len(w) > 1]

    @staticmethod
    def get_word_ngrams(tokens: List[str], n: int = 3) -> Set[Tuple[str, ...]]:
        """Extracts word n-grams."""
        if len(tokens) < n:
            return {tuple(tokens)} if tokens else set()
        return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

    @classmethod
    def compute_similarity(cls, text1: str, text2: str) -> Tuple[float, float, float]:
        """
        Computes (token_jaccard, ngram_jaccard, directional_containment).
        """
        toks1 = cls.get_word_tokens(text1)
        toks2 = cls.get_word_tokens(text2)
        if not toks1 or not toks2:
            return 0.0, 0.0, 0.0

        set1 = set(toks1)
        set2 = set(toks2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)
        token_jaccard = intersection / union if union > 0 else 0.0

        ngrams1 = cls.get_word_ngrams(toks1, n=3)
        ngrams2 = cls.get_word_ngrams(toks2, n=3)
        ng_intersection = len(ngrams1 & ngrams2)
        ng_union = len(ngrams1 | ngrams2)
        ngram_jaccard = ng_intersection / ng_union if ng_union > 0 else 0.0

        min_len = min(len(set1), len(set2))
        directional_containment = (intersection / min_len) if min_len > 0 else 0.0

        return token_jaccard, ngram_jaccard, directional_containment

    @classmethod
    def is_near_duplicate(
        cls,
        chunk1: EvidenceChunk,
        chunk2: EvidenceChunk,
        jaccard_threshold: float = 0.72,
        ngram_threshold: float = 0.60,
        containment_threshold: float = 0.85
    ) -> bool:
        """
        Determines whether two chunks are exact or near-duplicates.
        """
        norm1 = cls.normalize_text(chunk1.text)
        norm2 = cls.normalize_text(chunk2.text)

        # 1. Exact match after normalization
        if norm1 == norm2:
            return True

        # 2. Similarity metrics
        jaccard, ngram_jaccard, containment = cls.compute_similarity(chunk1.text, chunk2.text)

        if jaccard >= jaccard_threshold:
            return True

        if ngram_jaccard >= ngram_threshold:
            return True

        # 3. High containment with similar length ratio (e.g. sliding window chunk overlap)
        len1 = len(norm1.split())
        len2 = len(norm2.split())
        length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

        if containment >= containment_threshold and length_ratio >= 0.65:
            return True

        return False

    @classmethod
    def deduplicate(
        cls,
        evidence: EvidenceCollection,
        jaccard_threshold: float = 0.72,
        ngram_threshold: float = 0.60,
        containment_threshold: float = 0.85
    ) -> Tuple[EvidenceCollection, Dict[str, Any]]:
        """
        Deduplicates an EvidenceCollection, returning a clean collection with
        merged provenance and detailed deduplication diagnostics.
        """
        if not evidence.chunks or len(evidence.chunks) <= 1:
            diagnostics = {
                "original_count": len(evidence.chunks),
                "deduplicated_count": len(evidence.chunks),
                "duplicates_removed": 0,
                "clusters": []
            }
            return evidence, diagnostics

        clusters: List[List[EvidenceChunk]] = []

        for chunk in evidence.chunks:
            matched_cluster = None
            for cluster in clusters:
                rep = cluster[0]
                if cls.is_near_duplicate(
                    chunk,
                    rep,
                    jaccard_threshold=jaccard_threshold,
                    ngram_threshold=ngram_threshold,
                    containment_threshold=containment_threshold
                ):
                    matched_cluster = cluster
                    break

            if matched_cluster is not None:
                matched_cluster.append(chunk)
            else:
                clusters.append([chunk])

        deduplicated_chunks: List[EvidenceChunk] = []
        cluster_diagnostics: List[Dict[str, Any]] = []

        for rank_idx, cluster in enumerate(clusters, 1):
            # Select the highest quality representative (highest similarity score)
            best_chunk = max(cluster, key=lambda c: (c.similarity_score, -c.rank))
            duplicates = [c for c in cluster if c.chunk_id != best_chunk.chunk_id]

            # Merge provenance metadata
            merged_ids = list(dict.fromkeys([best_chunk.chunk_id] + [d.chunk_id for d in duplicates]))
            merged_sources = list(dict.fromkeys([best_chunk.source] + [d.source for d in duplicates]))
            merged_pages = list(dict.fromkeys(
                [p for p in ([best_chunk.page_number] + [d.page_number for d in duplicates]) if p is not None]
            ))

            # Shallow copy the representative to avoid mutating original retrieval state
            rep = copy.copy(best_chunk)
            meta = dict(rep.metadata or {})
            meta["merged_provenance"] = {
                "merged_chunk_ids": merged_ids,
                "merged_sources": merged_sources,
                "merged_pages": merged_pages,
                "duplicate_count": len(duplicates)
            }
            rep.metadata = meta
            rep.rank = rank_idx
            deduplicated_chunks.append(rep)

            if duplicates:
                cluster_diagnostics.append({
                    "representative_id": best_chunk.chunk_id,
                    "representative_source": best_chunk.source,
                    "duplicate_ids": [d.chunk_id for d in duplicates],
                    "duplicate_sources": [d.source for d in duplicates],
                    "merged_sources": merged_sources,
                    "duplicate_count": len(duplicates)
                })

        # Re-sort deduplicated chunks by similarity score
        deduplicated_chunks.sort(key=lambda c: -c.similarity_score)
        for i, c in enumerate(deduplicated_chunks, 1):
            c.rank = i

        diagnostics = {
            "original_count": len(evidence.chunks),
            "deduplicated_count": len(deduplicated_chunks),
            "duplicates_removed": len(evidence.chunks) - len(deduplicated_chunks),
            "clusters": cluster_diagnostics
        }

        # Build clean EvidenceCollection
        new_collection = EvidenceCollection(
            query=evidence.query,
            chunks=deduplicated_chunks,
            retriever_name=evidence.retriever_name,
            retrieval_time_ms=evidence.retrieval_time_ms,
            attractor_energy=evidence.attractor_energy,
            energy_depth=evidence.energy_depth,
            is_grounded=evidence.is_grounded,
            trajectory_hops=evidence.trajectory_hops
        )

        return new_collection, diagnostics
