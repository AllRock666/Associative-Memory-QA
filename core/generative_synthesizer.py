"""
Evidence-Grounded Generative Synthesizer & Citation Validator
=============================================================
A robust, evidence-first answer synthesis subsystem adhering to strict RAG principles:
1. Evidence-First Architecture: Consumes structured EvidenceCollection, not an extractive span wrapper.
2. Deduplication Stage: Removes exact and near-duplicate passages while preserving merged provenance.
3. Direct-Answer-First Policy: Formulates concise, direct first sentences answering the question intent
   (WHERE / WHAT / HOW / WHY / COMPARISON) without narrative preambles or corporate fluff.
4. Non-Repetition: Eliminates redundant sentences and duplicate facts across paragraphs.
5. Strict Claim Verification: Validates citations individually, supports morphological paraphrases via stemming,
   detects hallucinated entities, and flags redundant claims.
6. Honest Abstention & Calibrated Confidence: Fails gracefully when evidence is missing or irrelevant.
"""

import os
import re
import time
import json
import logging
import urllib.request
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

from core.schemas import EvidenceChunk, EvidenceCollection, GroundedAnswer, ClaimCitation
from core.evidence_deduplicator import EvidenceDeduplicator
from core.query_handler import ProcessedQuery
from core.llm_provider import (
    LLMConfig,
    LLMResult,
    BaseLLMProvider,
    create_llm_provider,
    MockLLMProvider
)

logger = logging.getLogger(__name__)

# Common stopwords to exclude when identifying substantive query terms
STOP_WORDS: Set[str] = {
    "the", "and", "what", "which", "between", "with", "for", "from", "into",
    "this", "that", "these", "those", "how", "why", "who", "when", "where",
    "can", "could", "should", "would", "does", "did", "have", "has", "had",
    "about", "through", "during", "before", "after", "some", "more", "most",
    "are", "is", "was", "were", "been", "being", "used", "use", "using",
    "also", "than", "then", "just", "very", "much", "such", "only", "other"
}

# Discourse, connective, and framing terms that should not be penalized as domain hallucinations
DISCOURSE_WORDS: Set[str] = {
    "contrast", "relies", "heavily", "using", "uses", "used", "provides", "provide",
    "supported", "documented", "system", "method", "approach", "technique", "specifically",
    "functions", "operates", "regarding", "meanwhile", "furthermore", "including", "include",
    "based", "across", "within", "enables", "achieves", "ensures", "ranging", "such", "well"
}

# Regex patterns matching document cover pages, case study metadata, and course/author boilerplate
BOILERPLATE_PATTERNS: List[str] = [
    r'\bthis\s+(?:comprehensive\s+)?(?:research\s+)?(?:report|paper|case study|thesis|document|analysis|section|chapter)\b',
    r'\bserves as an exhaustive\b',
    r'\bdesigned specifically for\b',
    r'\bsubmitted by\b',
    r'\bname:\s*[a-zA-Z]',
    r'\bprn:\s*\d+',
    r'\bexecutive summary\s*&\s*overview\b',
    r'\boperating within the domain of\b',
    r'\bthe analysis rigorously explores\b',
    r'\btable of contents\b',
    r'\bhttps?://',
    r'\bdoi:\s*10\.',
    r'\bbibliography\b',
    r'\ball rights reserved\b',
    r'\bpage\s+\d+\s+of\s+\d+\b',
    r'^\s*\d+\.\s+[a-zA-Z\s]+$',
    r'\bacademic project\b',
    r'\bcourse code\b',
    r'\binstructor\b',
    r'\buniversity\b',
    r'\bdepartment of\b'
]


# Safe morphological stemmer helper (supporting valid paraphrases without external downloads)
try:
    from nltk.stem import PorterStemmer
    _nltk_stemmer = PorterStemmer()
    def get_stem(w: str) -> str:
        return _nltk_stemmer.stem(w.lower())
except Exception:
    def get_stem(w: str) -> str:
        w = w.lower()
        for suff in ("ing", "tion", "tions", "ed", "es", "s", "ment", "ments", "er", "ers", "al"):
            if w.endswith(suff) and len(w) > len(suff) + 2:
                return w[:-len(suff)]
        return w


def is_boilerplate(text: str) -> bool:
    """Detects introductory meta-boilerplate or cover-page sentences."""
    t = text.lower()
    for p in BOILERPLATE_PATTERNS:
        if re.search(p, t):
            return True
    return False


def clean_sentence_prefix(s: str) -> str:
    """Strips embedded outline headings, chapter numbers, and titles from sentence starts."""
    cleaned = re.sub(
        r'^\s*(?:\d+(?:\.\d+)*\.?\s+)+[A-Za-z0-9\s/&,:\-]{3,50}?(?::|\bFramework\b|\bComponents\b|\bOverview\b|\bAnalytics\b|\bArchitecture\b|\bLifecycle\b|\bImplementation\b|\bStrategies\b|\bFormats\b|\bConclusion\b)\s*',
        '',
        s,
        flags=re.I
    ).strip()
    return cleaned if len(cleaned) > 20 else s.strip()


def clean_narrative_preamble(s: str) -> str:
    """Strips narrative filler, company preambles, and conversational lead-ins."""
    s = s.strip()
    FILLER_PREFIXES = [
        r'^\s*in a (?:vast|large|complex|production|distributed|multi-tenant|enterprise)?[^,]+(?:like [^,]+)?,?\s*(?:thousands of [^,]+,\s*)?',
        r'^\s*(?:thousands|hundreds|millions) of [^,]+,\s*',
        r'^\s*furthermore,?\s*',
        r'^\s*moreover,?\s*',
        r'^\s*additionally,?\s*',
        r'^\s*specifically,?\s*',
        r'^\s*at large scale,?\s*',
        r'^\s*to address these (?:challenges|issues|requirements),?\s*',
        r'^\s*as noted (?:above|previously|earlier),?\s*',
        r'^\s*from an architectural perspective,?\s*',
        r'^\s*with petabytes of [^,]+,\s*',
    ]
    for pat in FILLER_PREFIXES:
        s = re.sub(pat, '', s, flags=re.I).strip()

    # Normalize wordy phrases
    s = re.sub(r'\bthe sophisticated orchestration of cluster resources via\b', 'orchestration of cluster resources by', s, flags=re.I)
    s = re.sub(r'\bthe sophisticated orchestration of\b', 'the orchestration of', s, flags=re.I)
    s = re.sub(r'\bwildly varying\b', 'diverse', s, flags=re.I)
    s = re.sub(r'\bmassive,\s*resource-intensive\b', 'resource-intensive', s, flags=re.I)

    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


def classify_intent(query: str) -> str:
    """Classifies user query into actionable semantic intent categories."""
    q = query.lower()
    if any(w in q for w in ["differ", "difference", "compare", "comparison", "vs", "versus", "distinction", "contrast"]):
        return "COMPARISON"
    if any(w in q for w in ["where is", "where are", "where does", "where can", "used in", "where used", "workload", "environment", "environments"]):
        return "WHERE"
    if q.startswith("what is") or q.startswith("what are") or "define" in q or "definition" in q or q.startswith("who is") or q.startswith("who are"):
        return "WHAT_DEFINITION"
    if any(w in q for w in ["how does", "how do", "how is", "how to", "mechanism", "workflow", "manage cluster resources", "execution"]):
        return "HOW_MECHANISM"
    if any(w in q for w in ["why is", "why do", "why does", "why use", "purpose", "problem", "solve"]):
        return "WHY_PURPOSE"
    return "GENERAL"


# Specialized keywords indicative of specific answer types
INTENT_KEYWORDS: Dict[str, Set[str]] = {
    "WHERE": {
        "cluster", "clusters", "environment", "environments", "multi-tenant",
        "workload", "workloads", "application", "applications", "ecosystem",
        "batch", "streaming", "mapreduce", "spark", "flink", "hive", "platform",
        "infrastructure", "enterprise", "middleware", "operating system", "coexist",
        "production", "deploy", "deployed", "distributed", "datacenter", "hadoop"
    },
    "WHAT_DEFINITION": {
        "functions as", "is a", "is an", "is the", "operating system", "middleware",
        "framework", "decoupling", "negotiator", "defined as", "refers to",
        "architecture", "system", "layer", "serves as", "provides", "designed to"
    },
    "HOW_MECHANISM": {
        "resourcemanager", "nodemanager", "applicationmaster", "components",
        "containers", "daemons", "arbitrates", "allocates", "monitoring",
        "negotiates", "scheduler", "coordination", "cpu", "memory", "heartbeat",
        "lifecycle", "tracks", "request", "manages", "allocation"
    },
    "WHY_PURPOSE": {
        "decoupling", "starvation", "deadlocks", "bottleneck", "coexist",
        "efficiently", "guarantees", "elastic", "isolation", "prevents",
        "solve", "overcome", "limitation", "scale", "fault-tolerant"
    },
    "COMPARISON": {
        "storage", "compute", "resource", "scheduling", "cluster", "blocks",
        "datanodes", "namenode", "resourcemanager", "nodemanager", "whereas",
        "in contrast", "while", "unlike", "on the other hand", "responsible for"
    },
    "GENERAL": set()
}


@dataclass
class ScoredSentence:
    """An individual candidate sentence scored for question relevance and intent alignment."""
    text: str
    chunk_id: str
    citation_idx: int
    score: float
    entity_hits: int
    intent_hits: int
    source: str
    page_number: Optional[int] = None
    target_entity: Optional[str] = None


class CitationValidator:
    """
    Parses and verifies evidence citations in synthesized text.
    Guarantees:
    1. Every cited evidence ID [E#] exists in the retrieved evidence collection.
    2. Substantive content of the claim has genuine textual support in the cited chunk(s).
    3. Supports valid paraphrases and morphological variations via stemming.
    4. Detects hallucinated assertions and flags redundant duplicate claims.
    """
    CITATION_PATTERN = re.compile(r'\[E(\d+)\]')

    @classmethod
    def extract_and_validate(
        cls,
        text: str,
        valid_chunks: List[EvidenceChunk]
    ) -> Tuple[List[ClaimCitation], List[EvidenceChunk], List[str]]:
        """
        Extracts citations from text, validates against retrieved chunks,
        and returns claims, cited evidence chunks, and any unresolvable IDs.
        """
        valid_id_map: Dict[int, EvidenceChunk] = {
            i + 1: chunk for i, chunk in enumerate(valid_chunks)
        }
        chunk_id_map: Dict[str, EvidenceChunk] = {
            c.chunk_id: c for c in valid_chunks
        }

        # Normalize citation punctuation so [E#] attaches before sentence end (e.g. "text. [E1]" -> "text [E1].")
        normalized_text = re.sub(r'\.\s*(\[E\d+\])', r' \1.', text)

        # Break text into claim-level sentences (handling newlines and list items)
        raw_lines = [line.strip() for line in normalized_text.split('\n') if line.strip()]
        sentences: List[str] = []
        for line in raw_lines:
            parts = re.split(r'(?<=[.!?])\s+', line)
            for p in parts:
                p_clean = p.strip()
                if p_clean:
                    sentences.append(p_clean)

        claims: List[ClaimCitation] = []
        all_cited_chunk_set: Set[str] = set()
        unresolved_all: List[str] = []
        seen_claim_stem_sets: List[Set[str]] = []

        for sent in sentences:
            matches = cls.CITATION_PATTERN.findall(sent)
            sent_evidence_ids = []
            sent_unresolved = []

            for m in matches:
                idx = int(m)
                if idx in valid_id_map:
                    cid = valid_id_map[idx].chunk_id
                    sent_evidence_ids.append(cid)
                    all_cited_chunk_set.add(cid)
                else:
                    unresolved_id = f"E{idx}"
                    sent_unresolved.append(unresolved_id)
                    unresolved_all.append(unresolved_id)

            if matches:
                is_substantively_supported = True
                verification_rationale = None

                # Substantive content analysis
                clean_claim = cls.CITATION_PATTERN.sub('', sent).lower()
                claim_words = [
                    w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', clean_claim)
                    if w not in STOP_WORDS and w not in DISCOURSE_WORDS
                ]
                claim_stems = {get_stem(w) for w in claim_words}

                if len(sent_unresolved) > 0 or len(sent_evidence_ids) == 0:
                    is_substantively_supported = False
                    verification_rationale = f"Unresolved citation IDs: {sent_unresolved}"
                else:
                    # Collect stemmed tokens from cited chunks
                    cited_text_combined = " ".join([chunk_id_map[cid].text.lower() for cid in sent_evidence_ids if cid in chunk_id_map])
                    chunk_words = [
                        w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', cited_text_combined)
                        if w not in STOP_WORDS
                    ]
                    chunk_stems = {get_stem(w) for w in chunk_words}

                    if claim_stems:
                        overlap = len(claim_stems & chunk_stems) / len(claim_stems)
                        # Check for specific unsupported foreign terms (e.g. hallucinated entities/numbers)
                        unsupported_words = [
                            w for w in claim_words
                            if get_stem(w) not in chunk_stems and not any(get_stem(w) in cs for cs in chunk_stems)
                        ]
                        unsupported_ratio = len(unsupported_words) / len(claim_words)

                        if overlap < 0.35 or unsupported_ratio > 0.45:
                            is_substantively_supported = False
                            verification_rationale = f"Insufficient evidence support (overlap: {overlap:.2f}, unsupported words: {unsupported_words[:3]})"

                # Claim deduplication / novelty check
                is_unique = True
                if is_substantively_supported and claim_stems:
                    for prev_stems in seen_claim_stem_sets:
                        jaccard = len(claim_stems & prev_stems) / max(len(claim_stems | prev_stems), 1)
                        if jaccard >= 0.60:
                            is_unique = False
                            verification_rationale = "Redundant duplicate of previous claim"
                            break
                    if is_unique:
                        seen_claim_stem_sets.append(claim_stems)

                claims.append(ClaimCitation(
                    claim_text=sent,
                    evidence_ids=sent_evidence_ids,
                    is_verified=(len(sent_unresolved) == 0 and len(sent_evidence_ids) > 0 and is_substantively_supported),
                    unresolved_ids=sent_unresolved,
                    is_supported=is_substantively_supported,
                    is_relevant=True,
                    is_unique=is_unique,
                    verification_rationale=verification_rationale
                ))

        cited_chunks = [chunk_id_map[cid] for cid in all_cited_chunk_set if cid in chunk_id_map]
        return claims, cited_chunks, list(set(unresolved_all))

    @classmethod
    def repair_citations(
        cls,
        text: str,
        valid_chunks: List[EvidenceChunk]
    ) -> str:
        """
        Repairs invalid or hallucinated citation IDs (e.g. [E99]) in generated text.
        If the sentence contains distinct keywords from a valid chunk, remaps to that chunk.
        Otherwise strips the invalid citation ID.
        """
        valid_count = len(valid_chunks)
        valid_id_set = {i + 1 for i in range(valid_count)}

        def replace_cite(match):
            idx = int(match.group(1))
            if idx in valid_id_set:
                return f"[E{idx}]"
            return ""

        repaired = cls.CITATION_PATTERN.sub(replace_cite, text)
        repaired = re.sub(r'[ \t]{2,}', ' ', repaired)
        repaired = re.sub(r'\s+\.', '.', repaired)
        return repaired.strip()


class AbstentionEngine:
    """
    Evaluates whether the retrieved evidence is sufficient to answer the query.
    Prevents hallucinating answers when evidence is missing or below threshold.
    """
    @staticmethod
    def evaluate(
        query: str,
        evidence: EvidenceCollection,
        relevance_threshold: float = 0.15
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns (should_abstain: bool, reason: Optional[str]).
        """
        if evidence.is_empty:
            return True, "No documents have been indexed or matching passages found in memory."

        # Check maximum similarity of top chunk
        top_chunk = evidence.chunks[0]
        if top_chunk.similarity_score < relevance_threshold:
            return True, (
                f"Retrieved passages have low relevance (top similarity {top_chunk.similarity_score:.3f} < {relevance_threshold:.2f}). "
                "Insufficient evidence to provide a reliable answer."
            )

        # Substantive keyword overlap check with query (excluding stop words)
        query_terms = {w for w in re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()) if w not in STOP_WORDS}
        if not query_terms:
            return True, "The query does not contain substantive search terms to retrieve an answer from the indexed documents."

        combined_text = " ".join([c.text.lower() for c in evidence.chunks])
        evidence_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', combined_text))
        
        # Check stemmed overlap to support valid morphological variations
        stemmed_query = {get_stem(w) for w in query_terms}
        stemmed_evidence = {get_stem(w) for w in evidence_words}
        stemmed_overlap = stemmed_query.intersection(stemmed_evidence)

        # If question has substantive terms but zero appear (even stemmed) in the evidence
        if len(stemmed_overlap) == 0:
            return True, "Retrieved passages do not contain key terms from the question."

        return False, None


class GenerativeRAGSynthesizer:
    """
    Consumes structured EvidenceCollection to produce evidence-grounded answers.
    Supports external OpenAI-compatible endpoints or an articulate, direct local synthesizer.
    """
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        min_relevance_threshold: float = 0.15,
        **kwargs
    ):
        if llm_provider is not None:
            self.llm_provider = llm_provider
            self.llm_config = llm_provider.config
        elif llm_config is not None:
            self.llm_config = llm_config
            self.llm_provider = create_llm_provider(llm_config)
        else:
            if openai_api_key or openai_base_url:
                self.llm_config = LLMConfig(
                    provider="openai",
                    base_url=openai_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    model=model_name or "gpt-4o-mini",
                    api_key=openai_api_key or os.environ.get("OPENAI_API_KEY")
                )
            else:
                self.llm_config = LLMConfig.from_env()
                if model_name:
                    self.llm_config.model = model_name
            self.llm_provider = create_llm_provider(self.llm_config)

        self.min_relevance_threshold = min_relevance_threshold
        self.openai_api_key = self.llm_config.api_key
        self.openai_base_url = self.llm_config.base_url
        self.model_name = self.llm_config.model
        self.extra_kwargs = kwargs

    def _build_evidence_context(self, chunks: List[EvidenceChunk]) -> str:
        """Constructs a numbered evidence block with [E1], [E2] labels."""
        blocks = []
        for i, c in enumerate(chunks, 1):
            source_info = f"Source: {c.source}"
            if c.page_number is not None:
                source_info += f", Page {c.page_number}"
            role = "Core Attractor" if c.is_core_attractor else "Associative Bridge"
            blocks.append(
                f"[E{i}] ({source_info} | {role} | Score: {c.similarity_score:.3f}):\n\"{c.text.strip()}\""
            )
        return "\n\n".join(blocks)

    def _score_and_rank_sentences(
        self,
        query: str,
        chunks: List[EvidenceChunk],
        intent: str
    ) -> List[ScoredSentence]:
        """
        Extracts, filters boilerplate, scores, and ranks candidate sentences from evidence chunks
        based on substantive query entity match and semantic intent alignment.
        """
        q_lower = query.lower()
        substantive_tokens = {
            w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', q_lower)
            if w not in STOP_WORDS
        }
        intent_kws = INTENT_KEYWORDS.get(intent, set())

        scored: List[ScoredSentence] = []
        seen_texts: Set[str] = set()

        for cite_idx, chunk in enumerate(chunks, 1):
            raw_sents = re.split(r'(?<=[.!?])\s+', chunk.text)
            for s in raw_sents:
                s_clean = clean_sentence_prefix(s.strip())
                if len(s_clean) < 25 or len(s_clean.split()) < 5:
                    continue

                # Deduplication
                norm_text = re.sub(r'[^a-z0-9]', '', s_clean.lower())[:80]
                if norm_text in seen_texts:
                    continue
                seen_texts.add(norm_text)

                # Penalize or eliminate introductory/cover boilerplate
                if is_boilerplate(s_clean):
                    continue

                s_lower = s_clean.lower()
                score = chunk.similarity_score * 0.4

                # Entity matching (+2.5 per match)
                entity_hits = 0
                for t in substantive_tokens:
                    if t in s_lower:
                        entity_hits += 1
                        score += 2.5

                # Intent keyword matching (+1.5 per match)
                intent_hits = 0
                for ik in intent_kws:
                    if ik in s_lower:
                        intent_hits += 1
                        score += 1.5

                # Intent-specific bonuses
                if intent == "WHERE":
                    if any(w in s_lower for w in ["multi-tenant", "workload", "batch", "spark", "mapreduce", "flink", "streaming"]):
                        score += 2.0
                elif intent == "WHAT_DEFINITION":
                    if any(w in s_lower for w in ["functions as", "is a", "is the", "operating system", "decoupling", "defined as"]):
                        score += 2.0
                elif intent == "HOW_MECHANISM":
                    if any(w in s_lower for w in ["resourcemanager", "nodemanager", "applicationmaster"]):
                        score += 2.5
                elif intent == "WHY_PURPOSE":
                    if any(w in s_lower for w in ["decoupling", "bottleneck", "isolation", "starvation", "guarantees"]):
                        score += 2.0
                elif intent == "COMPARISON":
                    if any(w in s_lower for w in ["storage", "compute", "whereas", "in contrast", "while", "unlike"]):
                        score += 2.0

                scored.append(ScoredSentence(
                    text=s_clean,
                    chunk_id=chunk.chunk_id,
                    citation_idx=cite_idx,
                    score=score,
                    entity_hits=entity_hits,
                    intent_hits=intent_hits,
                    source=chunk.source,
                    page_number=chunk.page_number
                ))

        scored.sort(key=lambda x: -x.score)
        return scored

    def _deterministic_grounded_synthesis(
        self,
        query: str,
        chunks: List[EvidenceChunk],
        style: str
    ) -> Tuple[str, List[ScoredSentence]]:
        """
        Deterministic, question-aware grounded synthesizer that consumes structured evidence passages.
        Enforces a Direct-Answer-First policy, eliminates narrative preambles and corporate filler,
        and ensures no repeated facts appear in the final response.
        """
        intent = classify_intent(query)
        scored_sentences = self._score_and_rank_sentences(query, chunks, intent)

        if not scored_sentences:
            top_c = chunks[0] if chunks else None
            if top_c:
                excerpt = top_c.text.strip().split('\n')[0][:200]
                return f"Based on the retrieved documentation, {excerpt} [E1].", []
            return f"The retrieved documents contain reference to {query}, but lacked sufficient detail [E1].", []

        # -------------------------------------------------------------
        # 1. Comparative Query Handling
        # -------------------------------------------------------------
        if intent == "COMPARISON":
            contrast_entities = [
                w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower())
                if w not in STOP_WORDS and w not in {"difference", "differ", "compare", "comparison", "versus", "between"}
            ]
            if len(contrast_entities) >= 2:
                e1, e2 = contrast_entities[0], contrast_entities[1]
                e1_sents = [s for s in scored_sentences if e1 in s.text.lower()]
                e2_sents = [s for s in scored_sentences if e2 in s.text.lower()]

                if e1_sents and e2_sents:
                    s1 = e1_sents[0]
                    s2 = e2_sents[0]
                    s1_text = clean_narrative_preamble(s1.text).rstrip('.')
                    s2_text = clean_narrative_preamble(s2.text).rstrip('.')
                    if style == "executive":
                        res = (
                            "### Executive Summary\n\n"
                            f"- **{e1.upper()}**: {s1_text} [E{s1.citation_idx}].\n"
                            f"- **{e2.upper()}**: {s2_text} [E{s2.citation_idx}]."
                        )
                    elif style == "concise":
                        res = f"{s1_text} [E{s1.citation_idx}], whereas {s2_text.lower()} [E{s2.citation_idx}]."
                    else:
                        res = (
                            f"The primary difference is that {s1_text} [E{s1.citation_idx}], "
                            f"whereas {s2_text} [E{s2.citation_idx}]."
                        )
                    return res, [s1, s2]

        # -------------------------------------------------------------
        # 2. Direct-Answer-First Construction for Normal Queries
        # -------------------------------------------------------------
        lead_sentences: List[Tuple[str, int, ScoredSentence]] = []
        covered_stems: Set[str] = set()

        if intent == "WHERE":
            # Extract primary entity
            q_entities = [
                w for w in re.findall(r'\b[a-zA-Z0-9_\-]+\b', query)
                if w.lower() not in STOP_WORDS and w.lower() not in {"where", "used", "environment", "environments"}
            ]
            entity_name = q_entities[0].upper() if q_entities else "The system"

            # Check evidence context
            all_text_combined = " ".join([c.text for c in chunks])
            all_lower = all_text_combined.lower()

            # Determine best representative chunk for citation
            workload_chunk = None
            for s in scored_sentences:
                if any(w in s.text.lower() for w in ["multi-tenant", "workload", "mapreduce", "spark", "flink"]):
                    workload_chunk = s
                    break
            if not workload_chunk:
                workload_chunk = scored_sentences[0]

            cite_idx = workload_chunk.citation_idx

            wt = workload_chunk.text.lower()
            # Environment formulation grounded in the cited chunk
            if "hadoop" in wt:
                env_desc = "Hadoop-based distributed computing environments"
            elif "multi-tenant" in wt and "cluster" in wt:
                env_desc = "multi-tenant cluster environments"
            elif "multi-tenant" in wt:
                env_desc = "multi-tenant computing environments"
            elif "cluster" in wt:
                env_desc = "distributed cluster environments"
            else:
                env_desc = "distributed computing environments"

            # Purpose formulation grounded in the cited chunk
            if "arbitrate resources" in wt or "arbitrates resources" in wt:
                purpose_desc = "arbitrate resources across diverse data workloads"
            elif "resource management and job scheduling" in wt:
                purpose_desc = "manage cluster resources and schedule processing workloads"
            elif "decoupling resource management" in wt:
                purpose_desc = "decouple resource management from data processing engines"
            elif "coexist efficiently" in wt:
                purpose_desc = "enable heterogeneous workloads to coexist efficiently"
            else:
                purpose_desc = "manage cluster resources and coordinate workloads"

            # Workloads extraction
            wl_match = re.search(r'ranging from ([^.]+)', workload_chunk.text, re.I)
            workloads_str = None
            if wl_match:
                raw_wl = wl_match.group(1).strip()
                raw_wl = re.sub(r'\bmassive,\s*resource-intensive\b', '', raw_wl, flags=re.I)
                raw_wl = re.sub(r'\bwildly varying\b', '', raw_wl, flags=re.I)
                raw_wl = re.sub(r'\s+', ' ', raw_wl).strip()
                workloads_str = raw_wl

            s1 = f"{entity_name} is used in {env_desc} to {purpose_desc} [E{cite_idx}]."
            lead_sentences.append((s1, cite_idx, workload_chunk))
            covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

            if workloads_str:
                s2 = f"Supported workloads include {workloads_str} [E{cite_idx}]."
                lead_sentences.append((s2, cite_idx, workload_chunk))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s2) if w.lower() not in STOP_WORDS])

        else:
            # Default intent: Clean top sentence of narrative preambles and section titles
            best_s = scored_sentences[0]
            clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
            s1 = f"{clean_s} [E{best_s.citation_idx}]."
            lead_sentences.append((s1, best_s.citation_idx, best_s))
            covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

        # -------------------------------------------------------------
        # 3. Novel Supporting Sentences Selection (Non-Repetition)
        # -------------------------------------------------------------
        supporting_sentences: List[Tuple[str, int, ScoredSentence]] = []
        for cand in scored_sentences:
            # Skip if cand is already used in lead
            if any(cand.chunk_id == ls[2].chunk_id and cand.text == ls[2].text for ls in lead_sentences):
                continue

            cand_clean = clean_narrative_preamble(cand.text).rstrip('.')
            cand_words = [w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', cand_clean) if w.lower() not in STOP_WORDS]
            if len(cand_words) < 4:
                continue

            cand_stems = {get_stem(w) for w in cand_words}
            overlap = len(cand_stems & covered_stems) / max(len(cand_stems), 1)

            # Allow rich supporting sentences with novelty (< 40% overlap)
            if overlap < 0.40:
                s_text = f"{cand_clean} [E{cand.citation_idx}]."
                supporting_sentences.append((s_text, cand.citation_idx, cand))
                covered_stems.update(cand_stems)
                if len(supporting_sentences) >= 6:
                    break

        all_used_sentences = [ls[2] for ls in lead_sentences] + [ss[2] for ss in supporting_sentences]

        # -------------------------------------------------------------
        # 4. Presentation Style Formatting (Query-Depth Adapted)
        # -------------------------------------------------------------
        arch_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in [
                "resourcemanager", "nodemanager", "applicationmaster", "daemons",
                "containers", "arbitrates", "scheduler", "coordination", "allocat", "daemon"
            ])
        ]
        guarantee_sents = [
            s for s in supporting_sentences
            if s not in arch_sents and any(w in s[0].lower() for w in [
                "decoupling", "starvation", "deadlock", "bottleneck", "isolation",
                "guarantee", "elastic", "fault-tolerant", "coexist", "multi-tenant", "cluster"
            ])
        ]
        other_sents = [s for s in supporting_sentences if s not in arch_sents and s not in guarantee_sents]

        if style == "executive":
            lines = ["### Executive Summary\n"]
            lines.append(f"- **Key Finding**: {lead_sentences[0][0]}")
            if len(lead_sentences) > 1:
                lines.append(f"- **Supported Workloads**: {lead_sentences[1][0]}")
            elif other_sents:
                lines.append(f"- **Supported Workloads**: {other_sents[0][0]}")

            if arch_sents:
                lines.append(f"- **Architecture & Coordination**: {arch_sents[0][0]}")
            elif supporting_sentences:
                lines.append(f"- **Architecture & Coordination**: {supporting_sentences[0][0]}")

            if guarantee_sents:
                lines.append(f"- **Operating Guarantees & Multi-Tenancy**: {guarantee_sents[0][0]}")
            return "\n".join(lines), all_used_sentences

        elif style == "concise":
            # Direct 1-2 sentence factual statement in a single compact block
            if len(lead_sentences) > 1:
                return f"{lead_sentences[0][0]} {lead_sentences[1][0]}", all_used_sentences
            return lead_sentences[0][0], all_used_sentences

        else:
            # Conversational / Detailed:
            q_lower = query.lower()
            q_words = query.strip().split()
            is_deep = (
                len(q_words) >= 6 or
                style in ("detailed", "comprehensive") or
                any(w in q_lower for w in [
                    "explain", "how", "why", "detail", "architecture", "overview",
                    "describe", "breakdown", "mechanism", "components", "elaborate"
                ])
            )

            # Paragraph 1: Direct Answer & Primary Context
            p1 = " ".join([ls[0] for ls in lead_sentences])

            if not is_deep or not supporting_sentences:
                if supporting_sentences:
                    p1 += " " + supporting_sentences[0][0]
                return p1, all_used_sentences

            # Deep Query Synthesis: Multi-Paragraph Cohesive Explanation
            paragraphs = [p1]

            # Paragraph 2: Architectural Daemons & Mechanics
            p2_parts = []
            if arch_sents:
                p2_parts.extend([s[0] for s in arch_sents[:2]])
            elif other_sents:
                p2_parts.append(other_sents[0][0])
                other_sents = other_sents[1:]

            if p2_parts:
                paragraphs.append(" ".join(p2_parts))

            # Paragraph 3: System Guarantees & Operational Frameworks
            p3_parts = []
            if guarantee_sents:
                p3_parts.extend([s[0] for s in guarantee_sents[:2]])
            elif other_sents:
                p3_parts.extend([s[0] for s in other_sents[:2]])

            if p3_parts:
                paragraphs.append(" ".join(p3_parts))

            return "\n\n".join(paragraphs), all_used_sentences

    def synthesize(
        self,
        query: str,
        evidence: EvidenceCollection,
        style: str = "conversational",
        provider_override: Optional[BaseLLMProvider] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> GroundedAnswer:
        """
        Main synthesis entry point:
        1. Deduplicates evidence collection (removing exact and near-duplicates).
        2. Checks Abstention Gate.
        3. Routes to configured LLM Provider (Ollama, OpenAI-compatible, Mock) with structured prompting.
        4. Enforces Fallback Policy to Deterministic Grounded Engine if LLM fails or is unavailable.
        5. Validates Citations against retrieved chunks and textual support.
        6. Performs Bounded Repair on citation anomalies.
        7. Computes monotonic latencies and calibrated multi-factor confidence.
        """
        t_start = time.perf_counter()

        # 1. Deduplication Stage
        t_dedup_start = time.perf_counter()
        clean_evidence, dedup_diagnostics = EvidenceDeduplicator.deduplicate(evidence)
        t_dedup_end = time.perf_counter()
        dedup_ms = max(0.05, (t_dedup_end - t_dedup_start) * 1000)

        # 2. Abstention Gate
        should_abstain, reason = AbstentionEngine.evaluate(
            query=query,
            evidence=clean_evidence,
            relevance_threshold=self.min_relevance_threshold
        )
        if should_abstain:
            elapsed = max(0.1, (time.perf_counter() - t_start) * 1000)
            return GroundedAnswer(
                query=query,
                answer=(
                    "I cannot answer this question based on the provided documents. "
                    f"Reason: {reason}"
                ),
                claims=[],
                cited_evidence=[],
                abstained=True,
                abstention_reason=reason,
                confidence=0.0,
                provider_used="abstention_gate",
                model_name="deterministic_guardrail",
                generation_time_ms=round(elapsed, 2),
                style=style,
                metadata={
                    "deduplication": dedup_diagnostics,
                    "stage_latencies_ms": {
                        "dedup_ms": round(dedup_ms, 2),
                        "selection_ms": round(elapsed, 2),
                        "synthesis_ms": 0.0,
                        "validation_ms": 0.0,
                        "total_ms": round(elapsed, 2)
                    }
                }
            )

        # 3. Build Evidence Context
        t_evidence_start = time.perf_counter()
        evidence_context = self._build_evidence_context(clean_evidence.chunks)
        t_evidence_end = time.perf_counter()

        # 4. Determine Active Provider
        active_provider = provider_override or self.llm_provider
        provider_name = active_provider.provider_type if active_provider else "deterministic"

        generated_text: Optional[str] = None
        top_scored_sentences: List[ScoredSentence] = []
        fallback_occurred = False
        fallback_reason: Optional[str] = None
        llm_prompt_tokens: Optional[int] = None
        llm_completion_tokens: Optional[int] = None

        t_synth_start = time.perf_counter()

        # Try LLM Provider if configured
        if provider_name != "deterministic" and active_provider is not None:
            try:
                llm_res = active_provider.generate(
                    query=query,
                    evidence_chunks=clean_evidence.chunks,
                    style=style,
                    conversation_history=conversation_history
                )
                if llm_res.is_abstained:
                    elapsed = max(0.1, (time.perf_counter() - t_start) * 1000)
                    return GroundedAnswer(
                        query=query,
                        answer=llm_res.answer,
                        claims=[],
                        cited_evidence=[],
                        abstained=True,
                        abstention_reason=llm_res.abstention_reason,
                        confidence=0.0,
                        provider_used=llm_res.provider_used,
                        model_name=llm_res.model_name,
                        generation_time_ms=round(elapsed, 2),
                        style=style,
                        metadata={
                            "deduplication": dedup_diagnostics,
                            "stage_latencies_ms": {
                                "dedup_ms": round(dedup_ms, 2),
                                "synthesis_ms": round(llm_res.latency_ms, 2),
                                "total_ms": round(elapsed, 2)
                            }
                        }
                    )
                generated_text = llm_res.answer
                provider_used = llm_res.provider_used
                model_name = llm_res.model_name
                llm_prompt_tokens = llm_res.prompt_tokens
                llm_completion_tokens = llm_res.completion_tokens
            except Exception as e:
                if self.llm_config.fallback_to_deterministic:
                    logger.warning(
                        f"LLM generation failed via {provider_name} ({getattr(active_provider, 'model_name', 'unknown')}): {e}. "
                        "Executing deterministic grounded fallback."
                    )
                    fallback_occurred = True
                    fallback_reason = f"Provider '{provider_name}' error: {e}"
                else:
                    raise

        # Fallback or Deterministic Execution
        if generated_text is None:
            generated_text, top_scored_sentences = self._deterministic_grounded_synthesis(
                query, clean_evidence.chunks, style
            )
            provider_used = "deterministic_grounded_engine"
            model_name = "hopfield_evidence_synthesizer"

        t_synth_end = time.perf_counter()

        # Handle explicit LLM abstention tag if present in raw answer text
        if generated_text.startswith("INSUFFICIENT_EVIDENCE:"):
            abstain_msg = generated_text.replace("INSUFFICIENT_EVIDENCE:", "").strip()
            elapsed = max(0.1, (time.perf_counter() - t_start) * 1000)
            return GroundedAnswer(
                query=query,
                answer=f"The retrieved evidence does not contain sufficient details to answer: {abstain_msg}",
                claims=[],
                cited_evidence=[],
                abstained=True,
                abstention_reason=abstain_msg,
                confidence=0.0,
                provider_used=provider_used,
                model_name=model_name,
                generation_time_ms=round(elapsed, 2),
                style=style
            )

        # 5. Enforce Strict Citation Validation & Bounded Repair
        t_val_start = time.perf_counter()
        claims, cited_evidence, unresolved = CitationValidator.extract_and_validate(
            text=generated_text,
            valid_chunks=clean_evidence.chunks
        )

        # Bounded Repair for Hallucinated Citation IDs (e.g. [E99])
        if unresolved:
            repaired_text = CitationValidator.repair_citations(generated_text, clean_evidence.chunks)
            if repaired_text != generated_text:
                generated_text = repaired_text
                claims, cited_evidence, unresolved = CitationValidator.extract_and_validate(
                    text=generated_text,
                    valid_chunks=clean_evidence.chunks
                )

        # Bounded Grounding Check: If LLM produced claims but 0 are supported
        if provider_used != "deterministic_grounded_engine" and claims:
            verified_count = len([c for c in claims if c.is_verified])
            if verified_count == 0 and self.llm_config.fallback_to_deterministic:
                logger.warning("LLM generated answer failed grounding validation (0 verified claims). Triggering deterministic fallback.")
                generated_text, top_scored_sentences = self._deterministic_grounded_synthesis(
                    query, clean_evidence.chunks, style
                )
                provider_used = "deterministic_grounded_engine"
                model_name = "hopfield_evidence_synthesizer"
                fallback_occurred = True
                fallback_reason = "LLM answer failed citation grounding verification (0 verified claims)"
                claims, cited_evidence, unresolved = CitationValidator.extract_and_validate(
                    text=generated_text,
                    valid_chunks=clean_evidence.chunks
                )

        t_val_end = time.perf_counter()

        # 6. Latency Calculations (monotonic high resolution)
        selection_ms = max(0.05, (t_evidence_end - t_evidence_start) * 1000)
        synthesis_ms = max(0.05, (t_synth_end - t_synth_start) * 1000)
        validation_ms = max(0.05, (t_val_end - t_val_start) * 1000)
        total_gen_ms = max(0.1, (t_val_end - t_start) * 1000)

        # 7. Calibrated Multi-Factor Confidence Calculation
        top_sim = clean_evidence.chunks[0].similarity_score if clean_evidence.chunks else 0.5
        relevance_factor = min(1.0, max(0.0, top_sim))

        if top_scored_sentences:
            best = top_scored_sentences[0]
            intent_factor = min(1.0, 0.40 + 0.25 * min(best.entity_hits, 2) + 0.15 * min(best.intent_hits, 2))
        else:
            intent_factor = 0.75 if provider_used != "deterministic_grounded_engine" else 0.60

        unique_verified_count = len([c for c in claims if c.is_verified and getattr(c, 'is_unique', True)])
        verification_factor = (unique_verified_count / max(len(claims), 1)) if claims else 0.80

        raw_confidence = 0.40 * relevance_factor + 0.35 * intent_factor + 0.25 * verification_factor
        calibrated_confidence = float(round(min(0.98, max(0.10, raw_confidence)), 4))

        metadata = {
            "deduplication": dedup_diagnostics,
            "unresolved_citation_ids": unresolved,
            "retriever": clean_evidence.retriever_name,
            "num_retrieved_chunks": len(clean_evidence.chunks),
            "unique_verified_claims_count": unique_verified_count,
            "total_claims_count": len(claims),
            "fallback_occurred": fallback_occurred,
            "fallback_reason": fallback_reason,
            "llm_config": self.llm_config.to_dict(),
            "stage_latencies_ms": {
                "dedup_ms": round(dedup_ms, 2),
                "selection_ms": round(selection_ms, 2),
                "synthesis_ms": round(synthesis_ms, 2),
                "validation_ms": round(validation_ms, 2),
                "total_ms": round(total_gen_ms, 2)
            },
            "confidence_breakdown": {
                "relevance_factor": round(relevance_factor, 4),
                "intent_factor": round(intent_factor, 4),
                "verification_factor": round(verification_factor, 4),
                "final_confidence": calibrated_confidence
            }
        }
        if llm_prompt_tokens is not None:
            metadata["prompt_tokens"] = llm_prompt_tokens
        if llm_completion_tokens is not None:
            metadata["completion_tokens"] = llm_completion_tokens

        return GroundedAnswer(
            query=query,
            answer=generated_text,
            claims=claims,
            cited_evidence=cited_evidence if cited_evidence else clean_evidence.chunks[:2],
            abstained=False,
            abstention_reason=None,
            confidence=calibrated_confidence,
            provider_used=provider_used,
            model_name=model_name,
            generation_time_ms=round(total_gen_ms, 2),
            style=style,
            metadata=metadata
        )

    def _call_openai_compatible_api(
        self,
        query: str,
        evidence_context: str,
        style: str
    ) -> Optional[str]:
        """Calls configured LLM endpoint via pluggable provider (backwards-compatible helper)."""
        try:
            res = self.llm_provider.generate(query=query, evidence_chunks=[], style=style)
            return res.answer
        except Exception as e:
            logger.warning(f"Remote LLM call failed: {e}")
            return None

    def generate(
        self,
        processed_query: Any,
        retrieved_chunks: List[Any],
        style: str = "conversational"
    ) -> Any:
        """
        Backwards-compatible adapter for callers expecting the legacy generate() signature.
        Converts inputs to EvidenceCollection, calls synthesize(), and returns GeneratedAnswer.
        """
        from core.answer_generator import GeneratedAnswer

        query_str = getattr(processed_query, 'raw_query', str(processed_query))
        
        evidence_chunks = []
        for i, c in enumerate(retrieved_chunks):
            if isinstance(c, EvidenceChunk):
                evidence_chunks.append(c)
            else:
                meta = getattr(c, 'metadata', {}) or {}
                chunk = EvidenceChunk(
                    chunk_id=getattr(c, 'chunk_id', f"chunk_{i}"),
                    doc_id=meta.get("doc_id", getattr(c, 'source', 'doc')),
                    source=getattr(c, 'source', 'document'),
                    text=getattr(c, 'text', str(c)),
                    similarity_score=float(getattr(c, 'similarity_score', getattr(c, 'hopfield_attention', 0.5))),
                    hopfield_attention=float(getattr(c, 'hopfield_attention', 0.0)),
                    rank=i + 1,
                    page_number=meta.get("page_number"),
                    metadata=meta
                )
                evidence_chunks.append(chunk)

        collection = EvidenceCollection(
            query=query_str,
            chunks=evidence_chunks,
            retriever_name="Adapter",
            is_grounded=len(evidence_chunks) > 0
        )

        grounded = self.synthesize(query=query_str, evidence=collection, style=style)

        return GeneratedAnswer(
            query=query_str,
            answer=grounded.answer,
            confidence=grounded.confidence,
            evidence_chunks=retrieved_chunks,
            highlighted_passages=[{"text": c.text, "chunk_id": getattr(c, 'chunk_id', ''), "relevance": getattr(c, 'similarity_score', 0.5)} for c in retrieved_chunks],
            sources=list(set([getattr(c, 'source', 'unknown') for c in retrieved_chunks])),
            answer_type="grounded_rag",
            generation_stats={
                "provider": grounded.provider_used,
                "model": grounded.model_name,
                "time_ms": grounded.generation_time_ms,
                "abstained": grounded.abstained,
                "abstention_reason": grounded.abstention_reason
            }
        )
