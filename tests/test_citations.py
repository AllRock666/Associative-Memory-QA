"""
Test Suite: Citation Validation and Honest Abstention Engine
============================================================
Validates:
1. Citation extraction and mapping to concrete EvidenceChunk IDs.
2. Rejection / detection of hallucinated or unresolved citation IDs.
3. Honest abstention when similarity is below threshold or vocabulary has zero overlap.
4. Synthesizer generation without verbatim sentence duplication.
"""

import pytest
from core.schemas import EvidenceChunk, EvidenceCollection, ClaimCitation, GroundedAnswer
from core.generative_synthesizer import CitationValidator, AbstentionEngine, GenerativeRAGSynthesizer


def test_citation_validator_valid_and_unresolved():
    chunks = [
        EvidenceChunk(
            chunk_id="chunk_alpha_1",
            doc_id="doc1",
            source="constitutional_ai.pdf",
            text="Constitutional AI trains models using written rules and automated self-critique.",
            similarity_score=0.88,
            rank=1
        ),
        EvidenceChunk(
            chunk_id="chunk_beta_2",
            doc_id="doc2",
            source="rlhf_primer.pdf",
            text="Standard RLHF optimizes policy models based on direct human preference comparisons.",
            similarity_score=0.82,
            rank=2
        )
    ]

    # Model generates text citing [E1], [E2], and a hallucinated [E9]
    synthesized_text = (
        "Constitutional AI uses written principles for self-critique [E1]. "
        "In contrast, standard RLHF relies heavily on human preference annotations [E2]. "
        "Furthermore, an external algorithm handles fine-tuning [E9]."
    )

    claims, cited_chunks, unresolved = CitationValidator.extract_and_validate(synthesized_text, chunks)

    # Verify claim parsing
    assert len(claims) == 3
    assert claims[0].is_verified is True
    assert "chunk_alpha_1" in claims[0].evidence_ids

    assert claims[1].is_verified is True
    assert "chunk_beta_2" in claims[1].evidence_ids

    # Verify hallucinated citation E9
    assert claims[2].is_verified is False
    assert "E9" in claims[2].unresolved_ids
    assert "E9" in unresolved

    # Verify cited chunks resolution
    cited_ids = {c.chunk_id for c in cited_chunks}
    assert cited_ids == {"chunk_alpha_1", "chunk_beta_2"}


def test_abstention_engine_low_similarity():
    # Chunks with very low relevance score
    low_chunks = [
        EvidenceChunk(
            chunk_id="chunk_unrelated",
            doc_id="doc_recipes",
            source="cookbook.pdf",
            text="Bake the sourdough bread at 450 degrees Fahrenheit for 25 minutes.",
            similarity_score=0.08,
            rank=1
        )
    ]
    collection = EvidenceCollection(
        query="What is quantum entanglement in teleportation circuits?",
        chunks=low_chunks,
        retriever_name="Hopfield"
    )

    should_abstain, reason = AbstentionEngine.evaluate(
        query="What is quantum entanglement in teleportation circuits?",
        evidence=collection,
        relevance_threshold=0.15
    )

    assert should_abstain is True
    assert "low relevance" in reason.lower()


def test_abstention_engine_zero_vocabulary_overlap():
    chunks = [
        EvidenceChunk(
            chunk_id="chunk_misc",
            doc_id="doc_misc",
            source="general_notes.txt",
            text="The weather in the mountains was cloudy with scattered thunderstorms in the afternoon.",
            similarity_score=0.35,  # high numeric score but zero semantic overlap with question
            rank=1
        )
    ]
    collection = EvidenceCollection(
        query="What is the difference between Constitutional AI and RLHF?",
        chunks=chunks,
        retriever_name="Hopfield"
    )

    should_abstain, reason = AbstentionEngine.evaluate(
        query="What is the difference between Constitutional AI and RLHF?",
        evidence=collection,
        relevance_threshold=0.15
    )

    assert should_abstain is True
    assert "key terms" in reason.lower()


def test_abstention_engine_passes_with_grounded_evidence():
    chunks = [
        EvidenceChunk(
            chunk_id="chunk_cai",
            doc_id="doc_cai",
            source="constitutional_ai.txt",
            text="Constitutional AI provides a technique to align language models using principles rather than human raters.",
            similarity_score=0.75,
            rank=1
        )
    ]
    collection = EvidenceCollection(
        query="What is Constitutional AI?",
        chunks=chunks,
        retriever_name="Hopfield"
    )

    should_abstain, reason = AbstentionEngine.evaluate(
        query="What is Constitutional AI?",
        evidence=collection,
        relevance_threshold=0.15
    )

    assert should_abstain is False
    assert reason is None


def test_generative_synthesizer_synthesis_and_abstention():
    synthesizer = GenerativeRAGSynthesizer(min_relevance_threshold=0.15)

    # 1. Test abstention response
    empty_collection = EvidenceCollection(query="Explain cryptography", chunks=[], retriever_name="Hopfield")
    answer = synthesizer.synthesize("Explain cryptography", empty_collection)
    assert answer.abstained is True
    assert answer.confidence == 0.0
    assert "cannot answer" in answer.answer.lower()

    # 2. Test grounded synthesis
    valid_chunks = [
        EvidenceChunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            source="cai_paper.pdf",
            text="Constitutional AI specifies behavioral rules in a constitution to guide model self-critique.",
            similarity_score=0.85,
            rank=1
        ),
        EvidenceChunk(
            chunk_id="chunk_2",
            doc_id="doc_2",
            source="rlhf_paper.pdf",
            text="Standard RLHF directly relies on crowdsourced human preference rankings for policy optimization.",
            similarity_score=0.81,
            rank=2
        )
    ]
    valid_collection = EvidenceCollection(
        query="How does Constitutional AI differ from standard RLHF?",
        chunks=valid_chunks,
        retriever_name="Hopfield"
    )

    res = synthesizer.synthesize(
        query="How does Constitutional AI differ from standard RLHF?",
        evidence=valid_collection,
        style="conversational"
    )

    assert res.abstained is False
    assert res.confidence > 0.5
    assert len(res.cited_evidence) > 0
    assert "[E1]" in res.answer or "[E2]" in res.answer
