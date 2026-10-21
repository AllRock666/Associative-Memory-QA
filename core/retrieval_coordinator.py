"""
Retrieval Coordinator & Evidence Sufficiency Engine
===================================================
Coordinates:
1. Document Hopfield retrieval.
2. Explicit multi-factor evidence sufficiency evaluation.
3. Persistent knowledge cache inspection.
4. Web research retrieval & extraction.
5. Incremental Hopfield web memory embedding & associative retrieval.
6. Unified evidence merging with strict source-type provenance.
"""

import time
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

from core.schemas import EvidenceChunk, EvidenceCollection
from core.embeddings import DocumentChunker, EmbeddingEngine, Chunk
from core.hopfield_torch import PyTorchHopfieldMemory, RetrievalResult
from core.knowledge_memory import PersistentKnowledgeMemory
from core.web_search import BaseWebSearchProvider, WebPageExtractor, WebSearchResult, WebResearchConfig
from core.evidence_deduplicator import EvidenceDeduplicator

logger = logging.getLogger(__name__)


@dataclass
class SufficiencyDecision:
    """Detailed evaluation of whether retrieved document evidence adequately answers the question."""
    is_sufficient: bool
    confidence_score: float
    max_doc_score: float
    covered_concepts: List[str]
    missing_concepts: List[str]
    reasons: List[str]
    web_search_recommended: bool


class SufficiencyEvaluator:
    """
    Evaluates evidence sufficiency using multi-factor criteria:
    - Semantic relevance score threshold.
    - Essential question entity coverage.
    - Explicit freshness or external research intent in query.
    - Depth requirements of requested format (e.g. 8-mark vs concise).
    """

    FRESHNESS_TRIGGERS: Set[str] = {
        "latest", "recent", "recently", "current", "currently", "today",
        "newest", "update", "2024", "2025", "2026", "news", "trend",
        "trends", "search the web", "web search", "online", "look up"
    }

    @classmethod
    def evaluate(
        cls,
        query: str,
        doc_chunks: List[EvidenceChunk],
        format_id: Optional[str] = None,
        relevance_threshold: float = 0.40
    ) -> SufficiencyDecision:
        q_lower = query.lower()
        q_tokens = [w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', q_lower) if w not in {
            "what", "where", "when", "which", "explain", "describe", "detail", "overview",
            "the", "and", "for", "with", "mark", "marks", "answer", "question",
            "how", "why", "who", "whom", "whose", "does", "did", "done", "doing",
            "can", "could", "would", "should", "will", "shall", "may", "might", "must",
            "are", "was", "were", "been", "being", "have", "has", "had", "having",
            "its", "their", "theirs", "this", "that", "these", "those", "from",
            "into", "onto", "upon", "about", "above", "below", "between", "through"
        }]

        reasons = []
        missing = []
        covered = []

        # 1. Check for explicit freshness / web search triggers
        has_freshness_intent = any(trigger in q_lower for trigger in cls.FRESHNESS_TRIGGERS)
        if has_freshness_intent:
            reasons.append("Query explicitly requests fresh, current, or online web information.")

        # 2. Check document chunk availability & top similarity score
        if not doc_chunks:
            return SufficiencyDecision(
                is_sufficient=False,
                confidence_score=0.0,
                max_doc_score=0.0,
                covered_concepts=[],
                missing_concepts=q_tokens,
                reasons=["No document chunks were retrieved from uploaded knowledge store."],
                web_search_recommended=True
            )

        max_score = max(c.similarity_score for c in doc_chunks)
        combined_doc_text = " ".join(c.text.lower() for c in doc_chunks)

        # 3. Check entity / concept coverage with morphological tolerance
        for token in q_tokens:
            token_clean = token.lower()
            stem = token_clean.rstrip("s").rstrip("es").rstrip("ing").rstrip("ed")
            if (token_clean in combined_doc_text or 
                (len(stem) >= 3 and stem in combined_doc_text) or
                any(token_clean in word or (len(stem) >= 3 and stem in word) for word in combined_doc_text.split())):
                covered.append(token)
            else:
                missing.append(token)

        coverage_ratio = len(covered) / max(len(q_tokens), 1)

        # 4. Check score threshold
        score_ok = max_score >= relevance_threshold

        # 5. Check format-specific depth requirements
        format_needs_depth = format_id in ("8_mark", "10_mark", "detailed", "comparison")
        if format_needs_depth and len(doc_chunks) < 2 and len(combined_doc_text.split()) < 80:
            reasons.append(f"Requested format '{format_id}' requires in-depth multi-aspect evidence, but uploaded documents contain only thin context.")

        if not score_ok:
            reasons.append(f"Top document similarity ({max_score:.3f}) falls below confidence threshold ({relevance_threshold:.3f}).")

        if coverage_ratio < 0.5 and q_tokens:
            reasons.append(f"Document evidence misses key query concepts: {', '.join(missing)}.")

        is_sufficient = (
            score_ok and
            coverage_ratio >= 0.5 and
            not has_freshness_intent and
            not (format_needs_depth and len(combined_doc_text.split()) < 80)
        )

        confidence = round(float(max_score * coverage_ratio), 4)

        return SufficiencyDecision(
            is_sufficient=is_sufficient,
            confidence_score=confidence,
            max_doc_score=round(max_score, 4),
            covered_concepts=covered,
            missing_concepts=missing,
            reasons=reasons,
            web_search_recommended=not is_sufficient
        )


@dataclass
class CoordinatorRetrievalResult:
    """Comprehensive result of unified document and web retrieval."""
    evidence_collection: EvidenceCollection
    sufficiency_decision: SufficiencyDecision
    web_search_performed: bool
    cached_knowledge_used: bool
    web_sources: List[Dict[str, Any]] = field(default_factory=list)
    web_chunks_count: int = 0
    doc_chunks_count: int = 0
    dedup_diagnostics: Dict[str, Any] = field(default_factory=dict)


class RetrievalCoordinator:
    """
    Coordinates multi-source retrieval across Uploaded Documents,
    Persistent Knowledge Memory, and Web Search with Hopfield associative memory.
    """

    def __init__(
        self,
        doc_retriever,
        embedder: EmbeddingEngine,
        chunker: DocumentChunker,
        knowledge_memory: PersistentKnowledgeMemory,
        web_search_provider: BaseWebSearchProvider,
        web_extractor: WebPageExtractor,
        web_memory: Optional[PyTorchHopfieldMemory] = None,
        web_config: Optional[WebResearchConfig] = None
    ):
        self.doc_retriever = doc_retriever
        self.embedder = embedder
        self.chunker = chunker
        self.knowledge_memory = knowledge_memory
        self.web_search_provider = web_search_provider
        self.web_extractor = web_extractor
        self.web_config = web_config or WebResearchConfig()
        from core.web_search import MockWebSearchProvider
        if isinstance(self.web_search_provider, MockWebSearchProvider) and not getattr(self.web_extractor, 'mock_provider', None):
            self.web_extractor.mock_provider = self.web_search_provider
        # Dedicated Hopfield memory for web chunks
        self.web_memory = web_memory or PyTorchHopfieldMemory(beta=8.0, n_iter=1)

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        format_id: Optional[str] = None,
        web_search_enabled: Optional[bool] = None,
        force_web_search: bool = False,
        doc_filter: Optional[str] = None,
    ) -> CoordinatorRetrievalResult:
        """
        Executes unified retrieval:
        1. Retrieve uploaded document chunks.
        2. Evaluate evidence sufficiency.
        3. If insufficient or forced, check knowledge cache or search the web.
        4. Ingest, chunk, embed web content into Hopfield web memory.
        5. Merge document and web chunks into an integrated, deduplicated collection.
        """
        web_enabled = self.web_config.enabled if web_search_enabled is None else web_search_enabled

        # Step 1: Retrieve uploaded document evidence
        doc_collection = self.doc_retriever.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            doc_filter=doc_filter,
        )
        doc_chunks = list(doc_collection.chunks)
        for c in doc_chunks:
            c.source_type = "document"

        # Step 2: Evaluate sufficiency
        decision = SufficiencyEvaluator.evaluate(
            query=query,
            doc_chunks=doc_chunks,
            format_id=format_id
        )

        web_search_performed = False
        cached_knowledge_used = False
        web_sources: List[Dict[str, Any]] = []
        retrieved_web_chunks: List[EvidenceChunk] = []

        # Step 3: Trigger Web Search or Knowledge Memory if needed
        should_web_search = web_enabled and (not decision.is_sufficient or force_web_search)

        if should_web_search:
            # Step 3a: Check Persistent Knowledge Memory for fresh matching chunks
            cached_chunks = self.knowledge_memory.search_knowledge(query, top_k=top_k)
            if cached_chunks and len(cached_chunks) >= 2 and not force_web_search:
                logger.info(f"Retrieved {len(cached_chunks)} chunks from Persistent Knowledge Memory for query '{query}'.")
                retrieved_web_chunks.extend(cached_chunks)
                cached_knowledge_used = True
                for c in cached_chunks:
                    web_sources.append({
                        "title": c.title or c.source,
                        "url": c.url,
                        "source": c.source,
                        "cached": True
                    })
            else:
                # Step 3b: Execute live or mock web search
                try:
                    search_results: List[WebSearchResult] = self.web_search_provider.search(
                        query=query,
                        max_results=self.web_config.max_search_results
                    )
                    if search_results:
                        web_search_performed = True
                        pages_to_fetch = search_results[:self.web_config.max_pages_to_fetch]

                        new_web_chunks: List[Chunk] = []
                        for sr in pages_to_fetch:
                            web_doc = self.web_extractor.fetch_and_extract(sr.url)
                            # Prefer extracted page text; fallback to search snippet if page was JS-rendered, blocked, or minimal
                            text_to_use = web_doc.text if (web_doc.text and len(web_doc.text) >= 50) else sr.snippet
                            title_to_use = web_doc.title if (web_doc.title and web_doc.title != web_doc.source_domain) else sr.title

                            if text_to_use and len(text_to_use) >= 20:
                                page_chunks = self.chunker.chunk_text(text_to_use, source=title_to_use)
                                for pc in page_chunks:
                                    pc.metadata["url"] = sr.url
                                    pc.metadata["title"] = title_to_use
                                    pc.metadata["published_date"] = web_doc.published_date or sr.published_date
                                    pc.metadata["source_domain"] = sr.source_domain
                                    pc.metadata["source_type"] = "web"
                                new_web_chunks.extend(page_chunks)

                                web_sources.append({
                                    "title": title_to_use,
                                    "url": sr.url,
                                    "source": sr.source_domain,
                                    "published_date": web_doc.published_date or sr.published_date,
                                    "cached": False
                                })

                        # Step 3c: Embed newly fetched web chunks
                        if new_web_chunks:
                            embedded_chunks = self.embedder.embed_chunks(new_web_chunks, show_progress=False)
                            patterns = np.array([c.embedding for c in embedded_chunks])
                            chunk_metas = [
                                {
                                    "chunk_id": c.chunk_id,
                                    "text": c.text,
                                    "source": c.source,
                                    "source_type": "web",
                                    "url": c.metadata.get("url"),
                                    "title": c.metadata.get("title"),
                                    "published_date": c.metadata.get("published_date")
                                }
                                for c in embedded_chunks
                            ]

                            # Store into Persistent Knowledge Memory
                            for c in embedded_chunks:
                                self.knowledge_memory.add_entry(
                                    text=c.text,
                                    source=c.source,
                                    url=c.metadata.get("url"),
                                    title=c.metadata.get("title"),
                                    published_date=c.metadata.get("published_date"),
                                    source_type="web"
                                )

                            # Incrementally add to Hopfield Web Memory
                            self.web_memory.add_patterns(patterns, chunk_metas)

                            # Retrieve top web chunks from Hopfield web memory
                            raw_web_res: List[RetrievalResult] = self.web_memory.retrieve(query_embedding, top_k=top_k)
                            for rank, wr in enumerate(raw_web_res, 1):
                                meta = wr.metadata or {}
                                retrieved_web_chunks.append(
                                    EvidenceChunk(
                                        chunk_id=wr.chunk_id,
                                        doc_id=meta.get("url") or wr.source,
                                        source=wr.source,
                                        text=wr.text,
                                        similarity_score=wr.similarity_score,
                                        hopfield_attention=wr.hopfield_attention,
                                        rank=rank,
                                        is_core_attractor=True,
                                        source_type="web",
                                        url=meta.get("url"),
                                        title=meta.get("title"),
                                        published_date=meta.get("published_date")
                                    )
                                )
                except Exception as e:
                    logger.warning(f"Web retrieval coordinator encountered error: {e}")

        # Step 4: Merge Document and Web Chunks
        combined_chunks: List[EvidenceChunk] = []
        combined_chunks.extend(doc_chunks)
        combined_chunks.extend(retrieved_web_chunks)

        # Sort by similarity score descending
        combined_chunks.sort(key=lambda c: c.similarity_score, reverse=True)

        # Deduplicate overlapping chunks
        merged_collection = EvidenceCollection(
            query=query,
            chunks=combined_chunks,
            retriever_name=f"UnifiedCoordinator({doc_collection.retriever_name}+WebHopfield)"
        )
        dedup_collection, dedup_diags = EvidenceDeduplicator.deduplicate(merged_collection)

        # Limit to bounded top-K
        final_k = max(top_k, 8) if should_web_search else top_k
        dedup_collection.chunks = dedup_collection.chunks[:final_k]

        return CoordinatorRetrievalResult(
            evidence_collection=dedup_collection,
            sufficiency_decision=decision,
            web_search_performed=web_search_performed,
            cached_knowledge_used=cached_knowledge_used,
            web_sources=web_sources,
            web_chunks_count=len(retrieved_web_chunks),
            doc_chunks_count=len(doc_chunks),
            dedup_diagnostics=dedup_diags
        )
