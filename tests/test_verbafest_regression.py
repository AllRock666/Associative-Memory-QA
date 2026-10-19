"""
Test Suite: VERBAFEST Regression, Relevance Validation, and Domain Isolation
=============================================================================
Guarantees:
1. "what is verbafest" produces accurate, grounded event details and zero cybersecurity/NLP contamination.
2. active_doc_id / doc_filter scoping strictly isolates retrieval to the selected document.
3. Unscoped Hopfield retrieval does not drift into foreign attractor basins.
4. Past-tense definition extraction works with numbered/TOC page section prefixes.
5. AnswerRelevanceValidator rejects cross-domain contaminated text.
6. Confidence is heavily penalized (<= 0.20) when generated answers are off-topic.
7. Multi-aspect queries on VERBAFEST cover Mind Saga, Group Discussion, and Debate.
8. Unrelated out-of-domain queries properly trigger honest abstention.
9. Non-CS event documents generate event-appropriate headings, not forced CS engineering titles.
10. Natural language marks/format requests ("for 5 marks", "for 2 marks") adapt correctly.
"""

import pytest
import numpy as np
from core.schemas import EvidenceChunk, EvidenceCollection, GroundedAnswer
from core.generative_synthesizer import (
    GenerativeRAGSynthesizer,
    AnswerRelevanceValidator,
    clean_sentence_prefix,
    extract_sentence_subject,
    classify_intent,
    extract_query_target
)
from core.associative_engine import HopfieldAssociativeEngine
from core.hopfield_torch import PyTorchHopfieldMemory
from core.retrievers import HopfieldRetrieverAdapter, DenseRetrieverAdapter
from core.llm_provider import MockLLMProvider


@pytest.fixture
def verbafest_evidence() -> EvidenceCollection:
    """Fixture providing authentic VERBAFEST 2026 event report evidence chunks."""
    chunks = [
        EvidenceChunk(
            chunk_id="c_vf_intro",
            doc_id="final report.pdf",
            source="final report.pdf",
            text=(
                "3 1. Introduction: VERBAFEST 2026 was a placement-oriented event organized by "
                "Team Mavericks with the objective of providing students practical exposure to the "
                "various stages of the recruitment and selection process followed by companies."
            ),
            similarity_score=0.88,
            rank=1,
            page_number=3,
            is_core_attractor=True
        ),
        EvidenceChunk(
            chunk_id="c_vf_competitions",
            doc_id="final report.pdf",
            source="final report.pdf",
            text=(
                "The event consisted of three major competitions: Mind Saga, Group Discussion and Debate. "
                "Each competition was designed to help students develop essential skills required during placements, "
                "including analytical thinking, communication, decision-making and problem-solving."
            ),
            similarity_score=0.84,
            rank=2,
            page_number=3,
            is_core_attractor=True
        ),
        EvidenceChunk(
            chunk_id="c_vf_mindsaga",
            doc_id="final report.pdf",
            source="final report.pdf",
            text=(
                "Mind Saga was conducted in three rounds: Aptitude Test, Gaming Round and Versant Test, "
                "focusing on logical reasoning, cognitive ability and communication skills. "
                "In VERBAFEST 2026, the Mind Saga competition was conducted through a dedicated website, "
                "replacing the traditional paper-based method used in previous years."
            ),
            similarity_score=0.81,
            rank=3,
            page_number=4,
            is_core_attractor=False
        ),
        EvidenceChunk(
            chunk_id="c_vf_gd_debate",
            doc_id="final report.pdf",
            source="final report.pdf",
            text=(
                "The Group Discussion segment included Normal GD, Abstract GD and Situational GD, enabling students "
                "to experience different discussion formats commonly used in placement processes. "
                "The Debate competition consisted of Normal Debate and Swap Pros and Cons rounds, encouraging "
                "participants to think critically and present their viewpoints confidently."
            ),
            similarity_score=0.79,
            rank=4,
            page_number=8,
            is_core_attractor=False
        ),
        EvidenceChunk(
            chunk_id="c_vf_mgmt_platform",
            doc_id="final report.pdf",
            source="final report.pdf",
            text=(
                "16 3. Special Features Management Website Implementation - Developer : Shivam Dombe. "
                "To ensure efficient coordination and smooth execution of the event, a dedicated event management "
                "platform was developed. This platform handled the overall workflow of VERBAFEST 2026, including "
                "participant registration, group formation, round management and evaluation through a centralized system."
            ),
            similarity_score=0.77,
            rank=5,
            page_number=16,
            is_core_attractor=False
        ),
        EvidenceChunk(
            chunk_id="c_vf_overview",
            doc_id="final report.pdf",
            source="final report.pdf",
            text=(
                "20 Final Overview: The main objective of VERBAFEST 2026 was to help students become more "
                "prepared for placement rounds by simulating actual recruitment processes. "
                "With a total of 180 participants, the event successfully achieved its goal of providing "
                "placement-oriented preparation in a structured and practical format."
            ),
            similarity_score=0.75,
            rank=6,
            page_number=20,
            is_core_attractor=False
        ),
    ]
    return EvidenceCollection(query="what is verbafest", chunks=chunks, retriever_name="HopfieldRetriever")


# ---------------------------------------------------------------------------
# Test 1: "what is verbafest" produces accurate event details and zero CS contamination
# ---------------------------------------------------------------------------
def test_verbafest_event_identity_synthesis(verbafest_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="what is verbafest",
        evidence=verbafest_evidence,
        style="detailed"
    )

    assert not res.abstained
    answer_lower = res.answer.lower()

    # Must accurately define VERBAFEST as a placement-oriented event by Team Mavericks
    first_sentence = res.answer.split('.')[0].lower()
    assert "verbafest 2026" in first_sentence
    assert "placement-oriented event" in first_sentence
    assert "team mavericks" in first_sentence

    # Strictly ZERO cybersecurity or prompt engineering buzzwords
    forbidden = [
        "cybersecurity", "threat detection", "incident response", "security logs",
        "forensic investigation", "security metrics", "prompt engineering",
        "chain-of-thought", "agentic", "resourcemanager", "nodemanager"
    ]
    for word in forbidden:
        assert word not in answer_lower, f"Forbidden contaminated term '{word}' found in answer: {res.answer}"

    # Must maintain high and honest confidence
    assert res.confidence >= 0.70
    assert len(res.claims) >= 2
    assert all(c.is_verified for c in res.claims)


# ---------------------------------------------------------------------------
# Test 2: doc_filter scoping isolates retrieval to the target document
# ---------------------------------------------------------------------------
def test_doc_filter_scoping_retrieval():
    memory = PyTorchHopfieldMemory(beta=8.0, n_iter=2, similarity_threshold=0.01)
    dim = 8
    p1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    p2 = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    p3 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    patterns = np.vstack([p1, p2, p3])

    metadata = [
        {"chunk_id": "vf_1", "doc_id": "final report.pdf", "source": "final report.pdf", "text": "VERBAFEST 2026 introduction"},
        {"chunk_id": "vf_2", "doc_id": "final report.pdf", "source": "final report.pdf", "text": "Mind Saga competition"},
        {"chunk_id": "sec_1", "doc_id": "security_telemetry.pdf", "source": "security_telemetry.pdf", "text": "Security logs and threat detection"}
    ]
    memory.store(patterns, metadata)

    engine = HopfieldAssociativeEngine(memory=memory, beta=8.0, max_hops=2)
    retriever = HopfieldRetrieverAdapter(engine)

    q_emb = np.array([0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Retrieval scoped strictly to final report.pdf
    scoped_res = retriever.retrieve("event competitions", query_embedding=q_emb, top_k=5, doc_filter="final report.pdf")
    assert len(scoped_res.chunks) == 2
    assert all(c.doc_id == "final report.pdf" or c.source == "final report.pdf" for c in scoped_res.chunks)

    # Retrieval scoped strictly to security_telemetry.pdf
    sec_res = retriever.retrieve("security logs", query_embedding=q_emb, top_k=5, doc_filter="security_telemetry.pdf")
    assert len(sec_res.chunks) == 1
    assert sec_res.chunks[0].doc_id == "security_telemetry.pdf" or sec_res.chunks[0].source == "security_telemetry.pdf"


# ---------------------------------------------------------------------------
# Test 3: Unscoped Hopfield retrieval does not drift across documents with target query
# ---------------------------------------------------------------------------
def test_unscoped_hopfield_no_cross_contamination():
    memory = PyTorchHopfieldMemory(beta=8.0, n_iter=2, similarity_threshold=0.01)
    dim = 4
    v_vf = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_sec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    patterns = np.vstack([v_vf, v_sec])

    metadata = [
        {"chunk_id": "vf_1", "doc_id": "final report.pdf", "source": "final report.pdf", "text": "VERBAFEST 2026 event report"},
        {"chunk_id": "sec_1", "doc_id": "security.pdf", "source": "security.pdf", "text": "Cybersecurity logs"}
    ]
    memory.store(patterns, metadata)

    engine = HopfieldAssociativeEngine(memory=memory, beta=8.0, max_hops=2)
    retriever = HopfieldRetrieverAdapter(engine)

    # Query aligned with VERBAFEST vector
    query_emb = np.array([0.98, 0.02, 0.0, 0.0], dtype=np.float32)
    res = retriever.retrieve("VERBAFEST event", query_embedding=query_emb, top_k=2)

    assert len(res.chunks) >= 1
    assert res.chunks[0].chunk_id == "vf_1"
    assert res.chunks[0].doc_id == "final report.pdf" or res.chunks[0].source == "final report.pdf"


# ---------------------------------------------------------------------------
# Test 4: Past-tense definition extraction and TOC prefix cleaning
# ---------------------------------------------------------------------------
def test_past_tense_definition_and_toc_prefix_cleaning():
    raw_toc_sent = (
        "1. Introduction 3 2.1 Mind Saga 4 2.2 Group Discussion 8 2.3 Debate 12 3 Special Features 16 4 Final Overview 20 "
        "3 1. Introduction: VERBAFEST 2026 was a placement-oriented event organized by Team Mavericks."
    )
    cleaned = clean_sentence_prefix(raw_toc_sent)
    assert cleaned.startswith("VERBAFEST 2026 was a placement-oriented event")

    subject = extract_sentence_subject(cleaned)
    assert subject == "verbafest 2026"

    # Single section prefix
    single_prefix = "3 1. Introduction: VERBAFEST 2026 was a placement-oriented event."
    cleaned_single = clean_sentence_prefix(single_prefix)
    assert cleaned_single == "VERBAFEST 2026 was a placement-oriented event."
    assert extract_sentence_subject(cleaned_single) == "verbafest 2026"


# ---------------------------------------------------------------------------
# Test 5: AnswerRelevanceValidator rejects domain-contaminated answers
# ---------------------------------------------------------------------------
def test_answer_relevance_validator_rejects_contamination(verbafest_evidence):
    contaminated_text = (
        "Security logs provide an essential foundation for threat detection, incident response, "
        "and forensic investigation by recording authentication attempts in cybersecurity systems."
    )
    is_rel, rel_score, diag = AnswerRelevanceValidator.validate(
        query="what is verbafest",
        answer_text=contaminated_text,
        evidence_chunks=verbafest_evidence.chunks
    )
    assert is_rel is False
    assert rel_score <= 0.20
    assert "fails to mention key query concepts" in diag.lower() or "cross-domain" in diag.lower()


# ---------------------------------------------------------------------------
# Test 6: Confidence is heavily penalized when generated answer is off-topic
# ---------------------------------------------------------------------------
def test_confidence_penalized_for_irrelevant_answer():
    synthesizer = GenerativeRAGSynthesizer()
    query = "what is the current stock price of Apple on NASDAQ?"
    unrelated_chunks = [
        EvidenceChunk(
            chunk_id="c_other",
            doc_id="report.pdf",
            source="report.pdf",
            text="The annual sports day was celebrated with football and basketball tournaments on the college grounds.",
            similarity_score=0.25,
            rank=1
        )
    ]
    coll = EvidenceCollection(query=query, chunks=unrelated_chunks, retriever_name="Hopfield")
    res = synthesizer.synthesize(query, coll)

    assert res.abstained is True
    assert res.confidence <= 0.20


# ---------------------------------------------------------------------------
# Test 7: Multi-aspect query on VERBAFEST covers all three major competitions
# ---------------------------------------------------------------------------
def test_verbafest_multi_aspect_competitions(verbafest_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="Explain the competitions conducted in VERBAFEST",
        evidence=verbafest_evidence,
        style="detailed"
    )

    assert not res.abstained
    ans_lower = res.answer.lower()
    # Must explicitly mention all 3 competitions
    assert "mind saga" in ans_lower
    assert "group discussion" in ans_lower
    assert "debate" in ans_lower
    # Must mention details of competitions
    assert any(w in ans_lower for w in ["aptitude", "versant", "gaming"])
    assert any(w in ans_lower for w in ["situational", "abstract", "normal"])


# ---------------------------------------------------------------------------
# Test 8: Insufficient evidence triggers honest abstention with zero hallucination
# ---------------------------------------------------------------------------
def test_insufficient_evidence_out_of_domain(verbafest_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="What is quantum phase estimation in Shor's algorithm?",
        evidence=verbafest_evidence,
        style="detailed"
    )

    assert res.abstained is True
    assert res.confidence == 0.0
    assert "quantum" in res.query.lower()


# ---------------------------------------------------------------------------
# Test 9: Non-CS event documents generate event-appropriate headings, not forced CS titles
# ---------------------------------------------------------------------------
def test_event_domain_appropriate_headings(verbafest_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="Describe VERBAFEST 2026 event architecture and features",
        evidence=verbafest_evidence,
        style="detailed"
    )

    assert not res.abstained
    # Should use event-appropriate headings
    assert any(h in res.answer for h in [
        "### Key Competitions & Event Structure",
        "### Digital Implementation & Special Features",
        "### Participation, Objectives & Final Outcomes"
    ])
    # Should NOT have forced CS architecture headings
    assert "### Structural Foundations & Core Principles" not in res.answer


# ---------------------------------------------------------------------------
# Test 10: Natural language format requests adapt without toolbar buttons
# ---------------------------------------------------------------------------
def test_natural_language_exam_formatting(verbafest_evidence):
    synthesizer = GenerativeRAGSynthesizer()

    # 2-mark query should be concise and direct
    res_2m = synthesizer.synthesize(
        query="What is VERBAFEST 2026? Answer for 2 marks.",
        evidence=verbafest_evidence,
        style="concise"
    )
    assert not res_2m.abstained
    words_2m = res_2m.answer.split()
    assert len(words_2m) < 100
    assert "verbafest 2026" in res_2m.answer.lower()

    # 8-mark query should have depth and multiple sections
    res_8m = synthesizer.synthesize(
        query="Explain VERBAFEST 2026 for 8 marks with competitions and features.",
        evidence=verbafest_evidence,
        style="detailed"
    )
    assert not res_8m.abstained
    assert "### " in res_8m.answer
    assert len(res_8m.claims) >= 3


# ---------------------------------------------------------------------------
# Test 11: 8-mark and 10-mark synthesis on VERBAFEST produce no daemon/cluster fallbacks
# ---------------------------------------------------------------------------
def test_verbafest_8mark_and_10mark_no_daemon_hallucinations(verbafest_evidence):
    synthesizer = GenerativeRAGSynthesizer()

    # 8-mark deterministic exam answer
    res_8m = synthesizer.synthesize(
        query="Explain VERBAFEST 2026 structure and competitions.",
        evidence=verbafest_evidence,
        style="8_mark"
    )
    assert not res_8m.abstained
    ans_lower_8m = res_8m.answer.lower()
    for forbidden in ["cluster master", "daemon", "container", "resourcemanager", "nodemanager", "multi-tenant"]:
        assert forbidden not in ans_lower_8m, f"Forbidden term '{forbidden}' found in 8-mark answer"
    assert "### 1. Event Overview & Core Purpose" in res_8m.answer
    assert "### 2. Key Competitions & Formats" in res_8m.answer

    # 10-mark deterministic exam answer with comparative matrix
    res_10m = synthesizer.synthesize(
        query="Provide a comprehensive breakdown of VERBAFEST 2026.",
        evidence=verbafest_evidence,
        style="10_mark"
    )
    assert not res_10m.abstained
    ans_lower_10m = res_10m.answer.lower()
    for forbidden in ["daemon", "slot partitioning", "scheduler tight coupling", "mapreduce", "spark", "flink"]:
        assert forbidden not in ans_lower_10m, f"Forbidden term '{forbidden}' found in 10-mark answer"
    assert "### 4. Comparative Competition Matrix" in res_10m.answer or "### 4. Comparative Synthesis Matrix" in res_10m.answer
    assert "| Dimension |" in res_10m.answer or "| Analysis Dimension |" in res_10m.answer


# ---------------------------------------------------------------------------
# Test 12: LLM partial hallucination is pruned while preserving verified claims
# ---------------------------------------------------------------------------
def test_llm_partial_hallucination_sentence_pruning(verbafest_evidence):
    mock = MockLLMProvider()
    # 2 verified claims and 1 hallucinated claim (66% verified >= 50%)
    mock_answer = (
        "### Overview\n"
        "VERBAFEST 2026 was organized by the Literary Club [E1]. "
        "Mind Saga was one of the core competitions held during the event [E2]. "
        "The cyber warfare unit deployed real-time malware analysis during the finals [E3]."
    )
    mock.set_custom_response(mock_answer)
    synthesizer = GenerativeRAGSynthesizer(llm_provider=mock)

    res = synthesizer.synthesize(
        query="What is VERBAFEST 2026?",
        evidence=verbafest_evidence,
        style="conversational"
    )

    assert not res.abstained
    # The hallucinated cyber warfare sentence must be pruned
    assert "cyber warfare" not in res.answer.lower()
    assert "malware" not in res.answer.lower()
    # The verified sentences must be preserved
    assert "literary club" in res.answer.lower()
    assert "mind saga" in res.answer.lower()
    # Verified claims count
    assert all(c.is_verified for c in res.claims)
    assert len(res.claims) == 2


# ---------------------------------------------------------------------------
# Test 13: Low LLM grounding (< 50% verified) triggers deterministic fallback
# ---------------------------------------------------------------------------
def test_llm_low_grounding_triggers_deterministic_fallback(verbafest_evidence):
    mock = MockLLMProvider()
    # 1 verified claim and 3 hallucinated claims (25% verified < 50%)
    mock_answer = (
        "VERBAFEST 2026 was organized by the Literary Club [E1]. "
        "The quantum computing cluster experienced significant thermal throttling [E2]. "
        "Autonomous drone swarms were deployed across the campus perimeter [E3]. "
        "A zero-day exploit compromised the core database infrastructure [E4]."
    )
    mock.set_custom_response(mock_answer)
    synthesizer = GenerativeRAGSynthesizer(llm_provider=mock)

    res = synthesizer.synthesize(
        query="What is VERBAFEST 2026?",
        evidence=verbafest_evidence,
        style="conversational"
    )

    assert not res.abstained
    # Fallback to deterministic synthesis should have occurred
    assert res.provider_used == "deterministic_grounded_engine"
    ans_lower = res.answer.lower()
    assert "quantum" not in ans_lower
    assert "drone" not in ans_lower
    assert "zero-day" not in ans_lower
    assert "exploit" not in ans_lower
    # Answer should be grounded in VERBAFEST evidence
    assert "verbafest" in ans_lower
    assert all(c.is_verified for c in res.claims)


# ---------------------------------------------------------------------------
# Test 14: Uncited substantive hallucinations are detected and pruned
# ---------------------------------------------------------------------------
def test_uncited_substantive_hallucination_detected_and_pruned(verbafest_evidence):
    mock = MockLLMProvider()
    # 2 verified cited claims and 1 uncited hallucinated sentence
    mock_answer = (
        "VERBAFEST 2026 was organized by the Literary Club [E1]. "
        "Mind Saga was one of the core competitions held during the event [E2]. "
        "A foreign adversarial intelligence group infiltrated the university networks during the festival."
    )
    mock.set_custom_response(mock_answer)
    synthesizer = GenerativeRAGSynthesizer(llm_provider=mock)

    res = synthesizer.synthesize(
        query="What is VERBAFEST 2026?",
        evidence=verbafest_evidence,
        style="conversational"
    )

    assert not res.abstained
    assert "foreign adversarial" not in res.answer.lower()
    assert "infiltrated" not in res.answer.lower()
    assert "literary club" in res.answer.lower()
    assert all(c.is_verified for c in res.claims)
