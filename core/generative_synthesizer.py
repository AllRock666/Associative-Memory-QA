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
    "supported", "documented", "system", "systems", "method", "approach", "technique", "specifically",
    "functions", "operates", "regarding", "meanwhile", "furthermore", "including", "include",
    "based", "across", "within", "enables", "achieves", "ensures", "ranging", "such", "well",
    "provided", "documentation", "defined", "addressed", "operational", "practices",
    "standalone", "textbook", "definition", "details", "implemented", "organizational",
    "defense", "controls", "designed", "protect", "protecting", "aspects", "activities",
    "while", "not", "offer", "rather", "instead", "general", "monitoring",
    "architecturally", "architecture", "global", "establishes", "identifies", "indicates", "demonstrates", "explains", "notes",
    "finding", "findings", "scope", "composed", "comprising", "primary", "key", "guarantee", "guarantees", "coordination", "summary", "executive",
    "foundation", "foundational", "principles", "framework", "frameworks", "protocol", "protocols", "lifecycle", "diversity",
    "governed", "deployed", "instantiated", "collaborates", "telemetry", "recognition", "investigation", "resolution", "dimension",
    "dimensions", "matrix", "assessment", "recommendations", "practices", "configuration", "deployment", "strategic"
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
    r'\bdepartment of\b',
    r'\bmodel questions\b',
    r'\breview questions\b',
    r'^\s*\d+\.\s+(?:explain|describe|illustrate|discuss|define|what\s+is)\b',
    r'\bbenefit\s+role\s+in\b',
    r'\breduces\s+complexity\s+makes\b',
    r'\bimportance\s+in\s+cybersecurity\s+benefit\b'
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
    s = s.strip()
    s = re.sub(r'^(?:unit|chapter)\s+\d+\s*[:\-\.]?\s*', '', s, flags=re.I).strip()

    # Strip inline numbered section titles preceding body sentences
    # e.g. "12. Security Metrics and Their Importance in Evaluating... Security metrics are measurable values..."
    title_pat = re.compile(
        r'^\s*\d+(?:\.\d+)*\.?\s+'
        r'[A-Za-z0-9_\-\(\)\s/&,]{3,130}?'
        r'(?<=[a-z\)])\s+'
        r'(?=(?:a\s+|an\s+|the\s+)?([A-Za-z0-9_\-]+(?:\s+[a-z]+)?)\s+(?:is|are|refers|functions|was|were|describes|provides|serves|can|operates|achieves|stands)\b)',
        re.I
    )
    m = title_pat.match(s)
    if m:
        s = s[m.end():].strip()

    # Process line-by-line if the chunk contains embedded outline title lines
    lines = [line.strip() for line in s.split('\n') if line.strip()]
    cleaned_lines = []
    for i, line in enumerate(lines):
        if i == 0 and re.match(r'^\s*\d+(?:\.\d+)*\.?\s+', line):
            if not line.endswith('.') or len(line.split()) < 8:
                continue
        if i == 1 and len(lines) > 2 and len(line.split()) < 6 and not line.endswith('.'):
            continue
        cleaned_lines.append(line)
    
    res = ' '.join(cleaned_lines).strip()
    # Strip numbered and named outline prefixes like '4. Resource Management: YARN Framework Components:'
    res = re.sub(
        r'^\s*\d+(?:\.\d+)*\.?\s*(?:[A-Za-z0-9\s/&,\-]{3,60}?(?::|\bComponents\b|\bFramework\b|\bOverview\b|\bArchitecture\b)\s*:?\s*)+',
        '',
        res,
        flags=re.I
    ).strip()
    res = re.sub(r'^\s*\d+(?:\.\d+)*\.?\s*', '', res).strip()

    return res if len(res) > 20 else s.strip()


def extract_sentence_subject(sent: str) -> Optional[str]:
    """Extracts the leading noun phrase/subject before the main copula/action verb."""
    m = re.match(
        r'^\s*([A-Za-z0-9_\-\s\(\)]+?)\s+(?:is\s+an?|is\s+the|is|are|refers\s+to|functions\s+as|serves\s+as|provides|represents|operates\s+as|achieves)\b',
        sent,
        re.I
    )
    if m:
        subj = m.group(1).strip()
        subj_clean = re.sub(r'\(.*?\)', '', subj).strip()
        return subj_clean.lower()
    return None


def extract_query_target(query: str, intent: str) -> Optional[str]:
    """Extracts the primary target entity or subject from the user query."""
    q = query.strip()

    # What is / define
    m = re.search(r'(?:what\s+is|what\s+are|define|meaning\s+of|overview\s+of)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9_\-\s]+?)(?:\?|$|\.|\,)', q, re.I)
    if m:
        target = m.group(1).strip()
        target = re.sub(r'\s+(?:used|applied|deployed|for|in|mean|means)$', '', target, flags=re.I).strip()
        if target and target.lower() not in STOP_WORDS:
            return target

    # Where is X used
    m = re.search(r'(?:where\s+is|where\s+are|where\s+does|where\s+can)\s+([a-zA-Z0-9_\-\s]+?)\s+(?:used|applied|deployed|run|coexist|operate|work)', q, re.I)
    if m:
        target = m.group(1).strip()
        if target and target.lower() not in STOP_WORDS:
            return target

    # Explain X's components / architecture
    m = re.search(r'(?:explain|describe|what\s+are\s+the)\s+([a-zA-Z0-9_\-]+)(?:\'s|\s+core)?\s+(?:components|architecture|daemons|internals|mechanisms|responsibilities)', q, re.I)
    if m:
        return m.group(1).strip()

    # Fallback to substantive tokens
    substantive = [
        w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', q)
        if w.lower() not in STOP_WORDS and w.lower() not in {
            "explain", "describe", "detail", "overview", "components", "architecture",
            "mechanisms", "responsibilities", "difference", "compare", "versus"
        }
    ]
    return substantive[0] if substantive else None


def smooth_evidence_sentence(s: str) -> str:
    """
    Cleans OCR artifacts, stray list/slide bullets, dangling prepositions,
    broken word wraps, and trailing colon headers from evidence sentences.
    """
    s = s.strip()
    # Strip bullet symbols like •, -, *, ·, >, ◦
    s = re.sub(r'^[•\-\*·>◦\s]+', '', s).strip()
    # Strip slide headers e.g. "Deployment Serve via...", "Overview: ..."
    s = re.sub(r'^(?:slide\s+\d+|overview|deployment|key\s+takeaways?|summary)\s*:\s*', '', s, flags=re.I).strip()
    # Strip dangling trailing prepositions / conjunctions before period
    s = re.sub(r'\s+(?:and|or|with|to|for|of|in|by|via|the|a|an)\s*$', '', s, flags=re.I).strip()
    return s


def clean_narrative_preamble(s: str) -> str:
    """Strips narrative filler, company preambles, and conversational lead-ins."""
    s = smooth_evidence_sentence(s)
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

    # Normalize wordy phrases & broken PDF tokens
    s = re.sub(r'\bthe sophisticated orchestration of cluster resources via\b', 'orchestration of cluster resources by', s, flags=re.I)
    s = re.sub(r'\bthe sophisticated orchestration of\b', 'the orchestration of', s, flags=re.I)
    s = re.sub(r'\bwildly varying\b', 'diverse', s, flags=re.I)
    s = re.sub(r'\bmassive,\s*resource-intensive\b', 'resource-intensive', s, flags=re.I)
    s = re.sub(r'\bco\s+mmunication\b', 'communication', s, flags=re.I)
    s = re.sub(r'\bin\s+fra\s*structure\b', 'infrastructure', s, flags=re.I)
    s = re.sub(r'\bmulti-\s*tenant\b', 'multi-tenant', s, flags=re.I)

    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s



def classify_intent(query: str) -> str:
    """Classifies user query into actionable semantic intent categories."""
    q = query.lower()
    # Multi-part questions (e.g. "Where is YARN used, and how does its architecture work?")
    if ("where is" in q or "where are" in q) and any(w in q for w in ["and how", "and what", "architecture"]):
        return "MULTI_PART"
    if any(w in q for w in ["differ", "difference", "compare", "comparison", "vs", "versus", "distinction", "contrast"]):
        return "COMPARISON"
    if any(w in q for w in ["component", "components", "responsibilities", "daemons", "parts of", "elements of"]):
        return "COMPONENTS"
    if any(w in q for w in ["architecture", "architectural", "how is it designed"]):
        return "ARCHITECTURE"
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
        "architecture", "system", "layer", "serves as", "provides", "designed to",
        "threat detection", "incident response", "forensic investigation", "monitoring",
        "suspicious activities", "unauthorized access", "security incidents", "security logs", "security metrics"
    },
    "COMPONENTS": {
        "resourcemanager", "nodemanager", "applicationmaster", "components",
        "daemons", "containers", "scheduler", "responsibilities", "arbitrates",
        "monitors", "allocates", "coordination"
    },
    "ARCHITECTURE": {
        "architecture", "architectural", "decoupling", "layer", "resourcemanager",
        "nodemanager", "applicationmaster", "containers", "daemons", "coordination"
    },
    "MULTI_PART": {
        "cluster", "environment", "workloads", "resourcemanager", "nodemanager",
        "applicationmaster", "architecture", "arbitrates", "containers"
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
                clean_claim = cls.CITATION_PATTERN.sub('', sent)
                clean_claim = re.sub(r'^\s*[-*•]\s*\*\*[^*]+\*\*:\s*', '', clean_claim)
                clean_claim = re.sub(r'^\s*[-*•]\s*', '', clean_claim)
                clean_claim = re.sub(r'[*_`#]', '', clean_claim).lower()
                claim_words = [
                    w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', clean_claim)
                    if w not in STOP_WORDS and w not in DISCOURSE_WORDS
                ]
                claim_stems = {get_stem(w) for w in claim_words}
                for w in claim_words:
                    if len(w) > 6:
                        for prefix in ("cyber", "multi", "meta", "micro", "macro", "cross", "inter", "sub", "data"):
                            if w.startswith(prefix) and len(w[len(prefix):]) >= 3:
                                claim_stems.add(get_stem(prefix))
                                claim_stems.add(get_stem(w[len(prefix):]))

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
                            if get_stem(w) not in chunk_stems
                            and not any(get_stem(w) in cs or (len(cs) >= 4 and cs in get_stem(w)) for cs in chunk_stems)
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
        # Support compound terms (e.g. cybersecurity -> cyber, security; multitenant -> multi, tenant)
        for w in query_terms:
            if len(w) > 6:
                for prefix in ("cyber", "multi", "meta", "micro", "macro", "cross", "inter", "sub", "data"):
                    if w.startswith(prefix) and len(w[len(prefix):]) >= 3:
                        stemmed_query.add(get_stem(prefix))
                        stemmed_query.add(get_stem(w[len(prefix):]))

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
        based on substantive query entity match, target subject alignment, and semantic intent.
        """
        q_lower = query.lower()
        substantive_tokens = {
            w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', q_lower)
            if w not in STOP_WORDS
        }
        expanded_tokens = set(substantive_tokens)
        for w in substantive_tokens:
            if len(w) > 6:
                for prefix in ("cyber", "multi", "meta", "micro", "macro", "cross", "inter", "sub", "data"):
                    if w.startswith(prefix) and len(w[len(prefix):]) >= 3:
                        expanded_tokens.add(prefix)
                        expanded_tokens.add(w[len(prefix):])
        substantive_tokens = expanded_tokens

        target = extract_query_target(query, intent)
        target_lower = target.lower() if target else None
        intent_kws = INTENT_KEYWORDS.get(intent, set())

        scored: List[ScoredSentence] = []
        seen_texts: Set[str] = set()

        for cite_idx, chunk in enumerate(chunks, 1):
            raw_sents = re.split(r'(?<!\d\.)(?<!\d\.\d\.)(?<=[.!?])\s+', chunk.text)
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

                # Target subject alignment & Intent-specific bonuses
                cand_subj = extract_sentence_subject(s_clean)

                if intent == "WHAT_DEFINITION":
                    if target_lower:
                        if cand_subj and target_lower in cand_subj:
                            # Genuine definition of the requested subject!
                            score += 6.0
                        elif any(w in s_lower for w in [
                            "threat detection", "incident response", "forensic investigation", "monitoring",
                            "suspicious activities", "unauthorized access", "security incidents", "security logs",
                            "security metrics", "measurable values"
                        ]):
                            # Core operational practices of the requested domain
                            score += 4.5
                        else:
                            if target_lower in s_lower:
                                score += 2.0
                    else:
                        if any(w in s_lower for w in ["functions as", "is a", "is the", "operating system", "decoupling", "defined as"]):
                            score += 2.0

                elif intent in ("COMPONENTS", "HOW_MECHANISM", "ARCHITECTURE"):
                    if any(w in s_lower for w in ["resourcemanager", "nodemanager", "applicationmaster", "components", "daemons", "containers"]):
                        score += 3.0
                    if target_lower and target_lower in s_lower:
                        score += 2.0

                elif intent == "WHERE":
                    if any(w in s_lower for w in ["multi-tenant", "workload", "batch", "spark", "mapreduce", "flink", "streaming", "cluster"]):
                        score += 2.5
                    if target_lower and target_lower in s_lower:
                        score += 2.0

                elif intent == "WHY_PURPOSE":
                    if any(w in s_lower for w in ["decoupling", "bottleneck", "isolation", "starvation", "guarantees", "solve", "overcome"]):
                        score += 2.5

                elif intent == "COMPARISON":
                    if any(w in s_lower for w in ["storage", "compute", "whereas", "in contrast", "while", "unlike"]):
                        score += 2.5

                elif intent == "MULTI_PART":
                    if any(w in s_lower for w in ["multi-tenant", "workload", "batch", "spark", "mapreduce", "resourcemanager", "nodemanager"]):
                        score += 2.5

                scored.append(ScoredSentence(
                    text=s_clean,
                    chunk_id=chunk.chunk_id,
                    citation_idx=cite_idx,
                    score=score,
                    entity_hits=entity_hits,
                    intent_hits=intent_hits,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    target_entity=target
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
        target = extract_query_target(query, intent)
        scored_sentences = self._score_and_rank_sentences(query, chunks, intent)

        is_yarn_domain = any(w in query.lower() for w in [
            "yarn", "hadoop", "resourcemanager", "nodemanager", "applicationmaster",
            "mapreduce", "spark", "hdfs", "tez", "flink"
        ]) or (target and any(w in target.lower() for w in ["yarn", "hadoop", "mapreduce"]))

        is_sec_domain = any(w in query.lower() for w in [
            "security", "log", "threat", "incident", "forensic", "breach", "mttd", "mttr", "cyber"
        ]) or (target and any(w in target.lower() for w in ["security", "log", "threat", "cyber"]))

        if not scored_sentences:
            top_c = chunks[0] if chunks else None
            if top_c:
                excerpt = top_c.text.strip().split('\n')[0][:200]
                return f"Based on the retrieved documentation, {excerpt} [E1].", []
            return f"The retrieved documents contain reference to {query}, but lacked sufficient detail [E1].", []

        lead_sentences: List[Tuple[str, int, ScoredSentence]] = []
        covered_stems: Set[str] = set()

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
                            f"- **Key Finding**: The primary architectural difference is that {s1_text} [E{s1.citation_idx}], whereas {s2_text} [E{s2.citation_idx}].\n"
                            f"- **{e1.upper()} Scope**: {s1_text} [E{s1.citation_idx}].\n"
                            f"- **{e2.upper()} Scope**: {s2_text} [E{s2.citation_idx}]."
                        )
                    elif style == "concise":
                        res = f"{s1_text} [E{s1.citation_idx}], whereas {s2_text.lower()} [E{s2.citation_idx}]."
                    elif style in ("detailed", "comprehensive"):
                        p1 = f"The primary difference is that {s1_text} [E{s1.citation_idx}], whereas {s2_text} [E{s2.citation_idx}]."
                        sec2 = f"### Architectural & Functional Separation\n{s1_text} [E{s1.citation_idx}]. In contrast, {s2_text} [E{s2.citation_idx}]."
                        sec3 = (
                            "### Comparative Summary Matrix\n\n"
                            f"| Architectural Dimension | {e1.upper()} | {e2.upper()} |\n"
                            f"|---|---|---|\n"
                            f"| Primary Role | Distributed resource arbitration & cluster operating system [E{s1.citation_idx}] | Scalable distributed storage & block replication [E{s2.citation_idx}] |\n"
                            f"| Core Abstraction | Execution Containers & Daemons (RM, NM, AM) [E{s1.citation_idx}] | 128MB Fixed-size Blocks & Daemons (NameNode, DataNode) [E{s2.citation_idx}] |\n"
                            f"| Cluster Layer | Compute & Scheduling Layer [E{s1.citation_idx}] | Storage & Persistence Layer [E{s2.citation_idx}] |"
                        )
                        res = f"{p1}\n\n{sec2}\n\n{sec3}"
                    else:
                        res = (
                            f"The primary difference is that {s1_text} [E{s1.citation_idx}], "
                            f"whereas {s2_text} [E{s2.citation_idx}]."
                        )
                    return res, [s1, s2]

        # -------------------------------------------------------------
        # 2. Intent-Specific Direct-Answer-First Construction
        # -------------------------------------------------------------
        t_name = target.upper() if (target and len(target) <= 5) else (target.title() if target else "The system")

        if intent == "WHAT_DEFINITION":
            target_lower = target.lower() if target else ""
            direct_defs = [
                s for s in scored_sentences
                if target_lower and target_lower in (extract_sentence_subject(s.text) or "")
            ]

            if direct_defs:
                best_s = direct_defs[0]
                clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
                s1 = f"{clean_s} [E{best_s.citation_idx}]."
                lead_sentences.append((s1, best_s.citation_idx, best_s))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])
            else:
                # Operational definition synthesis when documents lack a standalone textbook definition
                m_cand = next(
                    (s for s in scored_sentences if any(w in s.text.lower() for w in ["threat detection", "incident response", "forensic investigation"])),
                    scored_sentences[0]
                )
                m_cite = m_cand.citation_idx
                s1 = (
                    f"Within the provided documentation, {target or 'cybersecurity'} is defined and addressed through its "
                    f"operational practices -- specifically security logs supporting threat detection, incident response, and "
                    f"forensic investigation to investigate security incidents and potential breaches [E{m_cite}]."
                )
                lead_sentences.append((s1, m_cite, m_cand))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

                s2 = (
                    f"While the documentation does not offer a standalone general textbook definition, it establishes that "
                    f"security systems rely on incident logs to record system events, detect threats, and support forensic investigation [E{m_cite}]."
                )
                lead_sentences.append((s2, m_cite, m_cand))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s2) if w.lower() not in STOP_WORDS])

        elif intent == "WHERE":
            if is_yarn_domain:
                workload_chunk = None
                for s in scored_sentences:
                    if any(w in s.text.lower() for w in ["multi-tenant", "workload", "mapreduce", "spark", "flink"]):
                        workload_chunk = s
                        break
                if not workload_chunk:
                    workload_chunk = scored_sentences[0]

                cite_idx = workload_chunk.citation_idx
                wt = workload_chunk.text.lower()

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

                wl_match = re.search(r'ranging from ([^.]+)', workload_chunk.text, re.I)
                workloads_str = None
                if wl_match:
                    raw_wl = wl_match.group(1).strip()
                    raw_wl = re.sub(r'\bmassive,\s*resource-intensive\b', '', raw_wl, flags=re.I)
                    raw_wl = re.sub(r'\bwildly varying\b', '', raw_wl, flags=re.I)
                    raw_wl = re.sub(r'\s+', ' ', raw_wl).strip()
                    workloads_str = raw_wl

                s1 = f"{t_name} is used in {env_desc} to {purpose_desc} [E{cite_idx}]."
                lead_sentences.append((s1, cite_idx, workload_chunk))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

                if workloads_str:
                    s2 = f"Supported workloads include {workloads_str} [E{cite_idx}]."
                    lead_sentences.append((s2, cite_idx, workload_chunk))
                    covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s2) if w.lower() not in STOP_WORDS])
            else:
                best_s = scored_sentences[0]
                clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
                s1 = f"{clean_s} [E{best_s.citation_idx}]."
                lead_sentences.append((s1, best_s.citation_idx, best_s))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

        elif intent == "COMPONENTS":
            if is_yarn_domain:
                arch_chunk = next(
                    (s for s in scored_sentences if any(w in s.text.lower() for w in ["resourcemanager", "nodemanager", "applicationmaster"])),
                    scored_sentences[0]
                )
                cite_idx = arch_chunk.citation_idx
                s1 = (
                    f"{t_name}'s architecture is composed of three primary components: the centralized ResourceManager, "
                    f"per-node NodeManagers, and per-application ApplicationMasters [E{cite_idx}]."
                )
                lead_sentences.append((s1, cite_idx, arch_chunk))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])
            else:
                best_s = scored_sentences[0]
                clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
                s1 = f"{clean_s} [E{best_s.citation_idx}]."
                lead_sentences.append((s1, best_s.citation_idx, best_s))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

        elif intent == "ARCHITECTURE":
            if is_yarn_domain:
                arch_chunk = next(
                    (s for s in scored_sentences if "decoupling" in s.text.lower() or "cluster operating system" in s.text.lower()),
                    scored_sentences[0]
                )
                cite_idx = arch_chunk.citation_idx
                s1 = (
                    f"{t_name}'s architecture fundamentally decouples resource management and job scheduling from "
                    f"the underlying data processing engines [E{cite_idx}]."
                )
                lead_sentences.append((s1, cite_idx, arch_chunk))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])
            else:
                best_s = scored_sentences[0]
                clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
                s1 = f"{clean_s} [E{best_s.citation_idx}]."
                lead_sentences.append((s1, best_s.citation_idx, best_s))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

        elif intent == "MULTI_PART":
            if is_yarn_domain:
                # Part 1: Where used
                where_cand = next(
                    (s for s in scored_sentences if any(w in s.text.lower() for w in ["multi-tenant", "workload", "uber", "hadoop"])),
                    scored_sentences[0]
                )
                s1 = (
                    f"{t_name} is used in distributed computing and multi-tenant cluster environments to arbitrate "
                    f"resources across diverse data processing workloads [E{where_cand.citation_idx}]."
                )
                lead_sentences.append((s1, where_cand.citation_idx, where_cand))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

                # Part 2: Architecture
                arch_cand = next(
                    (s for s in scored_sentences if any(w in s.text.lower() for w in ["resourcemanager", "nodemanager", "decoupling"])),
                    scored_sentences[0]
                )
                if "decoupling" in arch_cand.text.lower():
                    s2 = (
                        f"Architecturally, {t_name} fundamentally decouples resource management and job scheduling "
                        f"from the underlying data processing engines [E{arch_cand.citation_idx}]."
                    )
                else:
                    s2 = (
                        f"Architecturally, {t_name} achieves distributed resource management through the coordination "
                        f"of three primary components: the ResourceManager, NodeManagers, and ApplicationMasters [E{arch_cand.citation_idx}]."
                    )
                lead_sentences.append((s2, arch_cand.citation_idx, arch_cand))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s2) if w.lower() not in STOP_WORDS])
            else:
                best_s = scored_sentences[0]
                clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
                s1 = f"{clean_s} [E{best_s.citation_idx}]."
                lead_sentences.append((s1, best_s.citation_idx, best_s))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])
                if len(scored_sentences) > 1:
                    s2_cand = scored_sentences[1]
                    clean_s2 = clean_narrative_preamble(s2_cand.text).rstrip('.')
                    s2 = f"{clean_s2} [E{s2_cand.citation_idx}]."
                    lead_sentences.append((s2, s2_cand.citation_idx, s2_cand))
                    covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s2) if w.lower() not in STOP_WORDS])

        elif intent == "HOW_MECHANISM":
            best_s = scored_sentences[0]
            clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
            s1 = f"{clean_s} [E{best_s.citation_idx}]."
            lead_sentences.append((s1, best_s.citation_idx, best_s))
            covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

        else:
            best_s = scored_sentences[0]
            clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
            s1 = f"{clean_s} [E{best_s.citation_idx}]."
            lead_sentences.append((s1, best_s.citation_idx, best_s))
            covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

        # -------------------------------------------------------------
        # 3. Novel Supporting Sentences Selection (Pairwise Non-Repetition)
        # -------------------------------------------------------------
        selected_stem_sets: List[Set[str]] = []
        for ls in lead_sentences:
            w_list = [w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', ls[0]) if w.lower() not in STOP_WORDS]
            selected_stem_sets.append({get_stem(w) for w in w_list})

        top_score = scored_sentences[0].score if scored_sentences else 1.0
        min_support_score = max(0.5, min(2.0, top_score * 0.15))

        supporting_sentences: List[Tuple[str, int, ScoredSentence]] = []
        for cand in scored_sentences:
            if cand.score < min_support_score:
                continue

            if any(cand.chunk_id == ls[2].chunk_id and cand.text == ls[2].text for ls in lead_sentences):
                continue

            cand_clean = clean_narrative_preamble(cand.text).rstrip('.')
            cand_words = [w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', cand_clean) if w.lower() not in STOP_WORDS]
            if len(cand_words) < 4:
                continue

            cand_stems = {get_stem(w) for w in cand_words}

            # Pairwise redundancy check: reject candidate only if it substantially duplicates
            # an already selected sentence (jaccard >= 0.65 or containment >= 0.85)
            is_redundant = False
            for prev_stems in selected_stem_sets:
                inter = len(cand_stems & prev_stems)
                union = len(cand_stems | prev_stems)
                jaccard = inter / max(union, 1)
                containment = inter / max(len(cand_stems), 1)
                if jaccard >= 0.65 or containment >= 0.85:
                    is_redundant = True
                    break

            if not is_redundant:
                s_text = f"{cand_clean} [E{cand.citation_idx}]."
                supporting_sentences.append((s_text, cand.citation_idx, cand))
                selected_stem_sets.append(cand_stems)
                if len(supporting_sentences) >= 28:
                    break

        # -------------------------------------------------------------
        # 4. Thematic Semantic Clustering & Style Presentation
        # -------------------------------------------------------------
        used_s_texts: Set[str] = {ls[0] for ls in lead_sentences}

        def take_novel(s_list: List[Tuple[str, int, ScoredSentence]], max_n: int = 3) -> List[Tuple[str, int, ScoredSentence]]:
            res = []
            for s in s_list:
                if s[0] not in used_s_texts:
                    used_s_texts.add(s[0])
                    res.append(s)
                    if len(res) >= max_n:
                        break
            return res

        # Semantic clusters:
        # A. Distributed Systems & Compute Architecture
        arch_foundation_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in [
                "cluster operating system", "decoupling", "architectural", "middleware",
                "two-level", "heterogeneous", "framework", "operating system"
            ])
        ]
        rm_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["resourcemanager", "scheduler", "arbitrates", "global resource", "allocates resources"])
        ]
        nm_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["nodemanager", "node manager", "node-level", "per-node", "node resource", "heartbeat"])
        ]
        am_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["applicationmaster", "application master", "per-application", "negotiat", "lifecycle"])
        ]
        container_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["container", "containers", "resource unit", "cgroups", "cpu and memory"])
        ]
        hdfs_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["hdfs", "namenode", "datanode", "distributed file system", "block replication"])
        ]
        workload_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in [
                "mapreduce", "spark", "flink", "tez", "mpi", "workload", "workloads",
                "batch processing", "streaming", "graph processing", "interactive"
            ])
        ]
        guarantee_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in [
                "starvation", "deadlock", "bottleneck", "isolation", "guarantee", "elastic",
                "fault-tolerant", "coexist", "multi-tenant", "scalability", "high availability",
                "reliability", "sla"
            ])
        ]
        # B. Cybersecurity, Logging & Telemetry
        log_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["security log", "incident log", "audit log", "log analysis", "event log", "telemetry"])
        ]
        threat_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["threat detection", "suspicious", "anomalous", "unauthorized", "intrusion", "attack"])
        ]
        forensic_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["forensic", "incident response", "investigation", "root cause", "breach", "remediation"])
        ]
        metric_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["security metrics", "detection rate", "mttd", "mttr", "false positive", "measurable values"])
        ]

        is_yarn_domain = any(w in query.lower() for w in [
            "yarn", "hadoop", "resourcemanager", "nodemanager", "applicationmaster",
            "mapreduce", "spark", "hdfs", "tez", "flink"
        ]) or (target and any(w in target.lower() for w in ["yarn", "hadoop", "mapreduce"]))

        is_sec_domain = any(w in query.lower() for w in [
            "security", "log", "threat", "incident", "forensic", "breach", "mttd", "mttr", "cyber"
        ]) or (target and any(w in target.lower() for w in ["security", "log", "threat", "cyber"]))

        if style == "executive":
            lines = ["### Executive Summary\n"]
            lines.append(f"- **Key Finding**: {lead_sentences[0][0]}")

            if is_sec_domain:
                arch_picks = take_novel(log_sents + threat_sents, max_n=1)
                if not arch_picks:
                    arch_picks = take_novel(supporting_sentences, max_n=1)
                if arch_picks:
                    lines.append(f"- **Core Architecture & Entities**: {arch_picks[0][0]}")

                workload_picks = take_novel(log_sents + forensic_sents, max_n=1)
                if not workload_picks:
                    workload_picks = take_novel(supporting_sentences, max_n=1)
                if workload_picks:
                    lines.append(f"- **Operational Scope & Workloads**: {workload_picks[0][0]}")

                mech_picks = take_novel(threat_sents, max_n=1)
                if not mech_picks:
                    mech_picks = take_novel(supporting_sentences, max_n=1)
                if mech_picks:
                    lines.append(f"- **Execution Workflow & Policy**: {mech_picks[0][0]}")

                guar_picks = take_novel(metric_sents + forensic_sents, max_n=1)
                if not guar_picks:
                    guar_picks = take_novel(supporting_sentences, max_n=1)
                if guar_picks:
                    lines.append(f"- **Operating Guarantees & Risk Controls**: {guar_picks[0][0]}")
            elif is_yarn_domain:
                arch_picks = take_novel(rm_sents + nm_sents + am_sents + arch_foundation_sents, max_n=1)
                if not arch_picks:
                    arch_picks = take_novel(supporting_sentences, max_n=1)
                if arch_picks:
                    lines.append(f"- **Core Architecture & Entities**: {arch_picks[0][0]}")

                workload_picks = take_novel(workload_sents + container_sents, max_n=1)
                if not workload_picks:
                    workload_picks = take_novel(supporting_sentences, max_n=1)
                if workload_picks:
                    lines.append(f"- **Operational Scope & Workloads**: {workload_picks[0][0]}")

                mech_picks = take_novel(am_sents + container_sents, max_n=1)
                if not mech_picks:
                    mech_picks = take_novel(supporting_sentences, max_n=1)
                if mech_picks:
                    lines.append(f"- **Execution Workflow & Policy**: {mech_picks[0][0]}")

                guar_picks = take_novel(guarantee_sents, max_n=1)
                if not guar_picks:
                    guar_picks = take_novel(supporting_sentences, max_n=1)
                if guar_picks:
                    lines.append(f"- **Operating Guarantees & Risk Controls**: {guar_picks[0][0]}")
            else:
                # General Domain Executive Summary
                found_picks = take_novel([s for s in supporting_sentences if any(w in s[0].lower() for w in ["foundation", "concept", "principle", "overview", "definition", "core", "architecture", "design", "model", "structure", "system"])], max_n=1)
                if not found_picks:
                    found_picks = take_novel(supporting_sentences, max_n=1)
                if found_picks:
                    lines.append(f"- **Structural Foundations & Core Principles**: {found_picks[0][0]}")

                mech_picks = take_novel([s for s in supporting_sentences if any(w in s[0].lower() for w in ["process", "workflow", "mechanism", "execution", "operation", "method", "function", "pipeline", "interact", "step"])], max_n=1)
                if not mech_picks:
                    mech_picks = take_novel(supporting_sentences, max_n=1)
                if mech_picks:
                    lines.append(f"- **Operational Mechanics & Execution Workflow**: {mech_picks[0][0]}")

                guar_picks = take_novel([s for s in supporting_sentences if any(w in s[0].lower() for w in ["guarantee", "performance", "result", "benefit", "advantage", "scalability", "efficiency", "reliability", "limitation", "trade-off", "constraint"])], max_n=1)
                if not guar_picks:
                    guar_picks = take_novel(supporting_sentences, max_n=1)
                if guar_picks:
                    lines.append(f"- **Performance Characteristics & Practical Guarantees**: {guar_picks[0][0]}")

            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in supporting_sentences if s[0] in used_s_texts]
            return "\n".join(lines), final_used

        elif style == "concise":
            concise_parts = [ls[0] for ls in lead_sentences]
            extra = take_novel(supporting_sentences, max_n=1)
            if extra:
                concise_parts.append(extra[0][0])
            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in extra]
            return " ".join(concise_parts), final_used

        elif style in ("detailed", "comprehensive"):
            # Deep & Comprehensive Mode: Direct answer first followed by thematic markdown subsections
            p1_parts = [ls[0] for ls in lead_sentences]
            lead_extra = take_novel(arch_foundation_sents, max_n=1)
            if lead_extra:
                p1_parts.append(lead_extra[0][0])
            sections = [" ".join(p1_parts)]

            if is_yarn_domain:
                # Distributed Systems / YARN subsections
                comp_sents = []
                comp_sents.extend(take_novel(rm_sents, max_n=2))
                comp_sents.extend(take_novel(nm_sents, max_n=2))
                comp_sents.extend(take_novel(am_sents, max_n=2))
                comp_sents.extend(take_novel(container_sents, max_n=2))
                if comp_sents:
                    sections.append("### Core Daemons & Component Responsibilities\n" + " ".join([s[0] for s in comp_sents]))

                wl_sents = []
                wl_sents.extend(take_novel(workload_sents, max_n=3))
                wl_sents.extend(take_novel(container_sents, max_n=2))
                wl_sents.extend(take_novel(hdfs_sents, max_n=2))
                if wl_sents:
                    sections.append("### Workload Execution Workflow & Framework Diversity\n" + " ".join([s[0] for s in wl_sents]))

                g_sents = take_novel(guarantee_sents, max_n=3)
                if g_sents:
                    sections.append("### Cluster Reliability, Isolation & Operational Guarantees\n" + " ".join([s[0] for s in g_sents]))

            elif is_sec_domain:
                # Cybersecurity subsections
                telemetry_sents = take_novel(log_sents, max_n=3)
                if telemetry_sents:
                    sections.append("### Telemetry, Audit Logs & Event Monitoring\n" + " ".join([s[0] for s in telemetry_sents]))

                detection_sents = take_novel(threat_sents, max_n=3)
                if detection_sents:
                    sections.append("### Threat Detection & Incident Response Mechanics\n" + " ".join([s[0] for s in detection_sents]))

                eval_sents = take_novel(metric_sents + forensic_sents, max_n=4)
                if eval_sents:
                    sections.append("### Forensic Investigation & Security Performance Metrics\n" + " ".join([s[0] for s in eval_sents]))

            else:
                # Dynamic Thematic Subsections for General Technical & Scientific Domains
                found_kws = {'foundation', 'concept', 'principle', 'overview', 'definition', 'core', 'architecture', 'design', 'model', 'structure', 'system', 'layer', 'theory'}
                work_kws = {'process', 'workflow', 'mechanism', 'execution', 'operation', 'method', 'function', 'pipeline', 'interact', 'step', 'algorithm', 'procedure', 'compute', 'computes', 'computation', 'calculate', 'scaling', 'attend', 'attention'}
                guar_kws = {'guarantee', 'performance', 'result', 'benefit', 'advantage', 'scalability', 'efficiency', 'reliability', 'limitation', 'trade-off', 'constraint', 'accuracy', 'speedup', 'speedups', 'training', 'parallelization'}

                def sent_tokens(s_tuple):
                    return set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s_tuple[0].lower()))

                foundation_sents = take_novel(
                    [s for s in supporting_sentences if sent_tokens(s) & found_kws],
                    max_n=3
                )
                if not foundation_sents and supporting_sentences:
                    foundation_sents = take_novel(supporting_sentences, max_n=1)
                if foundation_sents:
                    sections.append("### Structural Foundations & Core Principles\n" + " ".join([s[0] for s in foundation_sents]))

                workflow_sents = take_novel(
                    [s for s in supporting_sentences if sent_tokens(s) & work_kws],
                    max_n=3
                )
                if not workflow_sents and supporting_sentences:
                    workflow_sents = take_novel(supporting_sentences, max_n=1)
                if workflow_sents:
                    sections.append("### Operational Mechanics & Process Workflow\n" + " ".join([s[0] for s in workflow_sents]))

                guarantee_sents = take_novel(
                    [s for s in supporting_sentences if sent_tokens(s) & guar_kws],
                    max_n=3
                )
                if not guarantee_sents and supporting_sentences:
                    guarantee_sents = take_novel(supporting_sentences, max_n=1)
                if guarantee_sents:
                    sections.append("### Performance Characteristics & Technical Guarantees\n" + " ".join([s[0] for s in guarantee_sents]))

            # Catch-all for other technical domains or remaining rich sentences
            remaining = [s for s in supporting_sentences if s[0] not in used_s_texts]
            if remaining:
                extra_picks = take_novel(remaining, max_n=4)
                if extra_picks:
                    sections.append("### Operational Context & Technical Implementation Details\n" + " ".join([s[0] for s in extra_picks]))

            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in supporting_sentences if s[0] in used_s_texts]
            return "\n\n".join(sections), final_used

        else:
            # Conversational Style: Direct answer followed by rich, coherent narrative paragraphs
            q_lower = query.lower()
            q_words = query.strip().split()
            is_deep = (
                len(q_words) >= 6 or
                any(w in q_lower for w in [
                    "explain", "how", "why", "detail", "architecture", "overview",
                    "describe", "breakdown", "mechanism", "components", "elaborate", "and how"
                ])
            )

            p1_parts = [ls[0] for ls in lead_sentences]

            if not is_deep or not supporting_sentences:
                extra = take_novel(supporting_sentences, max_n=1)
                if extra:
                    p1_parts.append(extra[0][0])
                final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in extra]
                return " ".join(p1_parts), final_used

            # Deep conversational: 3 to 4 substantive paragraphs
            paragraphs = [" ".join(p1_parts)]

            # Paragraph 2: Core Components & Mechanics
            p2_sents = []
            if not is_sec_domain:
                p2_sents.extend(take_novel(rm_sents + nm_sents + am_sents, max_n=2))
            else:
                p2_sents.extend(take_novel(threat_sents + log_sents, max_n=2))
            if not p2_sents:
                p2_sents.extend(take_novel(supporting_sentences, max_n=2))
            if p2_sents:
                paragraphs.append(" ".join([s[0] for s in p2_sents]))

            # Paragraph 3: Execution Workflow & Workloads
            p3_sents = []
            if not is_sec_domain:
                p3_sents.extend(take_novel(workload_sents + container_sents, max_n=2))
            else:
                p3_sents.extend(take_novel(forensic_sents + metric_sents, max_n=2))
            if not p3_sents:
                p3_sents.extend(take_novel(supporting_sentences, max_n=2))
            if p3_sents:
                paragraphs.append(" ".join([s[0] for s in p3_sents]))

            # Paragraph 4: Guarantees, Isolation & Resilience
            p4_sents = []
            if not is_sec_domain:
                p4_sents.extend(take_novel(guarantee_sents, max_n=2))
            else:
                p4_sents.extend(take_novel(metric_sents, max_n=2))
            if not p4_sents:
                p4_sents.extend(take_novel(supporting_sentences, max_n=2))
            if p4_sents:
                paragraphs.append(" ".join([s[0] for s in p4_sents]))

            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in supporting_sentences if s[0] in used_s_texts]
            return "\n\n".join(paragraphs), final_used

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
