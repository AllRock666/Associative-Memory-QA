"""
Regression Test Suite: HopfieldQA Answer Synthesis, Relevance, and Grounding
=============================================================================
Tests:
- Test A: "where is yarn used" - Direct-answer-first sentence, eliminates boilerplate, includes workloads.
- Test B: "what is YARN" - Definitional lead and decoupling architecture.
- Test C: "how does YARN manage cluster resources" - Component coordination (RM, NM, AM).
- Test D: "what is the difference between HDFS and YARN" - Comparative contrast of storage vs compute.
- Test E: Out-of-domain query - Honest abstention with zero confidence.
- Test F: Citation integrity - Rejection of ungrounded claims and hallucinated IDs [E99].
- Test G: Answer styles - Verified formatting for conversational, executive, and concise styles.
- Test H: Monotonic timing & Empty evidence handling.
- Test I: Evidence deduplication - Exact & near-duplicates merged with provenance preserved.
- Test J: Retention of distinct passages on the same topic.
- Test K: Non-repetition enforcement in synthesized output.
- Test L: Paraphrase verification via morphological stemming.
- Test M: Hopfield retrieval diagnostics and trajectory preservation.
"""

import pytest
from typing import List
from core.schemas import EvidenceChunk, EvidenceCollection, ClaimCitation, GroundedAnswer
from core.evidence_deduplicator import EvidenceDeduplicator
from core.generative_synthesizer import (
    GenerativeRAGSynthesizer,
    CitationValidator,
    AbstentionEngine,
    is_boilerplate,
    clean_sentence_prefix,
    clean_narrative_preamble,
    classify_intent
)


@pytest.fixture
def yarn_evidence_collection() -> EvidenceCollection:
    """Fixture providing realistic evidence chunks representing the Uber Big Data Case Study."""
    chunks = [
        # Chunk 1: Cover page / course boilerplate (simulating E1 in the bug report)
        EvidenceChunk(
            chunk_id="chunk_uber_cover",
            doc_id="uber_case_study.pdf",
            source="Uber Big Data Case Study.pdf",
            text=(
                "This comprehensive research report serves as an exhaustive case study designed specifically "
                "for the D1 Batch, submitted by student research analysts. Operating within the domain of "
                "distributed infrastructure, the analysis rigorously explores large-scale data workflows."
            ),
            similarity_score=0.8446,
            rank=1,
            page_number=1,
            is_core_attractor=True
        ),
        # Chunk 2: YARN definition
        EvidenceChunk(
            chunk_id="chunk_yarn_def",
            doc_id="uber_case_study.pdf",
            source="Uber Big Data Case Study.pdf",
            text=(
                "Yet Another Resource Negotiator (YARN) functions as the cluster operating system, "
                "fundamentally decoupling resource management and job scheduling from the underlying "
                "data processing engines."
            ),
            similarity_score=0.8329,
            rank=2,
            page_number=4,
            is_core_attractor=True
        ),
        # Chunk 3: Where YARN is used (workloads & multi-tenant environments)
        EvidenceChunk(
            chunk_id="chunk_yarn_workloads",
            doc_id="uber_case_study.pdf",
            source="Uber Big Data Case Study.pdf",
            text=(
                "In a vast, multi-tenant engineering environment like Uber's, YARN arbitrates resources across "
                "diverse data workloads, ranging from massive, resource-intensive batch MapReduce ETL jobs "
                "to interactive, low-latency Spark SQL queries and streaming Flink applications. "
                "YARN is the critical middleware enabling heterogeneous workloads to coexist efficiently."
            ),
            similarity_score=0.7995,
            rank=3,
            page_number=5,
            is_core_attractor=False
        ),
        # Chunk 4: How YARN manages resources (architecture)
        EvidenceChunk(
            chunk_id="chunk_yarn_arch",
            doc_id="uber_case_study.pdf",
            source="Uber Big Data Case Study.pdf",
            text=(
                "4. Resource Management: YARN Framework Components: YARN achieves distributed resource "
                "management through the coordination of three primary components: the ResourceManager (RM), "
                "NodeManagers (NM), and ApplicationMasters (AM). The ResourceManager arbitrates resources, "
                "while NodeManagers launch and monitor containers."
            ),
            similarity_score=0.5619,
            rank=4,
            page_number=5,
            is_core_attractor=False
        ),
        # Chunk 5: HDFS storage details (for comparison)
        EvidenceChunk(
            chunk_id="chunk_hdfs_storage",
            doc_id="uber_case_study.pdf",
            source="Uber Big Data Case Study.pdf",
            text=(
                "Hadoop Distributed File System (HDFS) provides scalable distributed storage across the cluster, "
                "dividing large files into fixed-size 128MB blocks replicated across DataNodes managed by a NameNode."
            ),
            similarity_score=0.5100,
            rank=5,
            page_number=3,
            is_core_attractor=False
        ),
    ]

    return EvidenceCollection(
        query="where is yarn used",
        chunks=chunks,
        retriever_name="HopfieldRetrieverAdapter"
    )


# ---------------------------------------------------------------------------
# Test A: Direct Answer First ("where is yarn used")
# ---------------------------------------------------------------------------
def test_regression_where_is_yarn_used(yarn_evidence_collection):
    synthesizer = GenerativeRAGSynthesizer()
    answer = synthesizer.synthesize("where is yarn used", yarn_evidence_collection, style="conversational")

    assert answer.abstained is False
    assert answer.confidence > 0.50

    # 1. Assert boilerplate and narrative fluff are completely excluded
    answer_lower = answer.answer.lower()
    assert "comprehensive research report" not in answer_lower
    assert "exhaustive case study" not in answer_lower
    assert "d1 batch" not in answer_lower
    assert not answer.answer.startswith("In a vast")

    # 2. Assert direct answer in the very first sentence
    assert answer.answer.startswith("YARN is used in")

    # 3. Assert environments and workloads are concisely detailed
    assert "multi-tenant" in answer_lower or "cluster" in answer_lower
    assert any(w in answer_lower for w in ["mapreduce", "spark", "flink", "workloads"])

    # 4. Assert proper citations are present and verified
    assert "[E3]" in answer.answer or "[E2]" in answer.answer or "[E4]" in answer.answer
    assert len(answer.claims) > 0
    assert all(c.is_verified for c in answer.claims)


# ---------------------------------------------------------------------------
# Test B: Definitional query ("what is YARN")
# ---------------------------------------------------------------------------
def test_regression_what_is_yarn(yarn_evidence_collection):
    synthesizer = GenerativeRAGSynthesizer()
    answer = synthesizer.synthesize("what is YARN", yarn_evidence_collection, style="conversational")

    assert answer.abstained is False
    answer_lower = answer.answer.lower()

    # Must prioritize definition directly in first sentence
    assert "comprehensive research report" not in answer_lower
    assert any(term in answer_lower for term in [
        "functions as", "operating system", "decoupling resource management", "resource negotiator"
    ])
    assert "[E2]" in answer.answer or "[E3]" in answer.answer or "[E4]" in answer.answer


# ---------------------------------------------------------------------------
# Test C: Mechanism query ("how does YARN manage cluster resources")
# ---------------------------------------------------------------------------
def test_regression_how_yarn_manages_resources(yarn_evidence_collection):
    synthesizer = GenerativeRAGSynthesizer()
    answer = synthesizer.synthesize(
        "how does YARN manage cluster resources",
        yarn_evidence_collection,
        style="conversational"
    )

    assert answer.abstained is False
    answer_lower = answer.answer.lower()

    # Must explain components directly
    assert "resourcemanager" in answer_lower
    assert "nodemanager" in answer_lower
    assert "[E4]" in answer.answer


# ---------------------------------------------------------------------------
# Test D: Comparison query ("what is the difference between HDFS and YARN")
# ---------------------------------------------------------------------------
def test_regression_difference_hdfs_and_yarn(yarn_evidence_collection):
    synthesizer = GenerativeRAGSynthesizer()
    answer = synthesizer.synthesize(
        "what is the difference between HDFS and YARN",
        yarn_evidence_collection,
        style="conversational"
    )

    assert answer.abstained is False
    answer_lower = answer.answer.lower()

    # Must contrast storage for HDFS and resource management for YARN
    assert "hdfs" in answer_lower
    assert "yarn" in answer_lower
    assert "storage" in answer_lower or "datanodes" in answer_lower
    assert "resource" in answer_lower or "operating system" in answer_lower


# ---------------------------------------------------------------------------
# Test E: Honest abstention on ungrounded query
# ---------------------------------------------------------------------------
def test_regression_honest_abstention_out_of_domain(yarn_evidence_collection):
    synthesizer = GenerativeRAGSynthesizer()
    query = "what is the current stock price of Apple on NASDAQ?"
    answer = synthesizer.synthesize(query, yarn_evidence_collection)

    assert answer.abstained is True
    assert answer.confidence == 0.0
    assert "cannot answer" in answer.answer.lower() or "insufficient" in answer.answer.lower()


# ---------------------------------------------------------------------------
# Test F: Strict citation validation & unsupported claim detection
# ---------------------------------------------------------------------------
def test_regression_citation_integrity(yarn_evidence_collection):
    chunks = yarn_evidence_collection.chunks

    # Case 1: Hallucinated citation index [E99]
    hallucinated_text = "YARN is a distributed framework [E99]."
    claims, cited_chunks, unresolved = CitationValidator.extract_and_validate(hallucinated_text, chunks)
    assert len(claims) == 1
    assert claims[0].is_verified is False
    assert "E99" in claims[0].unresolved_ids
    assert "E99" in unresolved

    # Case 2: Injected ungrounded claim citing [E2] with foreign terms not in E2
    ungrounded_claim = "The stock market crashed severely following the corporate acquisition [E2]."
    claims_ug, _, _ = CitationValidator.extract_and_validate(ungrounded_claim, chunks)
    assert len(claims_ug) == 1
    assert claims_ug[0].is_verified is False

    # Case 3: Grounded claim citing E2 with authentic substantive words
    grounded_claim = "YARN functions as the cluster operating system, decoupling resource management [E2]."
    claims_gr, _, _ = CitationValidator.extract_and_validate(grounded_claim, chunks)
    assert len(claims_gr) == 1
    assert claims_gr[0].is_verified is True


# ---------------------------------------------------------------------------
# Test G: Presentation styles (conversational, executive, concise)
# ---------------------------------------------------------------------------
def test_regression_answer_styles(yarn_evidence_collection):
    synthesizer = GenerativeRAGSynthesizer()

    # Executive style
    exec_answer = synthesizer.synthesize("where is yarn used", yarn_evidence_collection, style="executive")
    assert exec_answer.abstained is False
    assert "### Executive Summary" in exec_answer.answer
    assert "- **Key Finding**:" in exec_answer.answer
    assert "[E" in exec_answer.answer

    # Concise style
    concise_answer = synthesizer.synthesize("where is yarn used", yarn_evidence_collection, style="concise")
    assert concise_answer.abstained is False
    assert len(concise_answer.answer.split("\n\n")) == 1  # single compact paragraph
    assert "[E" in concise_answer.answer


# ---------------------------------------------------------------------------
# Test H: Monotonic timing & Empty evidence handling
# ---------------------------------------------------------------------------
def test_regression_monotonic_timing_and_empty_evidence():
    synthesizer = GenerativeRAGSynthesizer()

    # Empty evidence
    empty_col = EvidenceCollection(query="what is YARN", chunks=[], retriever_name="Hopfield")
    empty_res = synthesizer.synthesize("what is YARN", empty_col)
    assert empty_res.abstained is True
    assert empty_res.confidence == 0.0
    assert empty_res.generation_time_ms >= 0.05
    assert empty_res.metadata["stage_latencies_ms"]["total_ms"] >= 0.05


# ---------------------------------------------------------------------------
# Test I: Evidence Deduplication (Exact & Near-Duplicate Chunks)
# ---------------------------------------------------------------------------
def test_regression_evidence_deduplication_exact_and_near_duplicates():
    c1 = EvidenceChunk(
        chunk_id="c_workloads_1",
        doc_id="uber_case_study.pdf",
        source="doc_a.pdf",
        text="YARN arbitrates resources across diverse data workloads, ranging from batch MapReduce ETL jobs to interactive Spark SQL queries and streaming Flink applications.",
        similarity_score=0.88,
        rank=1,
        page_number=5
    )
    # Near duplicate with high token overlap from sliding window chunking
    c2 = EvidenceChunk(
        chunk_id="c_workloads_2",
        doc_id="uber_case_study.pdf",
        source="doc_b.pdf",
        text="YARN arbitrates resources across diverse data workloads, ranging from batch MapReduce ETL jobs to interactive Spark SQL queries and streaming Flink applications. YARN is the critical middleware.",
        similarity_score=0.86,
        rank=2,
        page_number=6
    )
    col = EvidenceCollection(query="where is yarn used", chunks=[c1, c2], retriever_name="Hopfield")
    dedup_col, diagnostics = EvidenceDeduplicator.deduplicate(col)

    assert diagnostics["duplicates_removed"] == 1
    assert len(dedup_col.chunks) == 1
    rep = dedup_col.chunks[0]
    assert rep.chunk_id == "c_workloads_1"
    # Merged provenance metadata preserved
    merged_prov = rep.metadata["merged_provenance"]
    assert "c_workloads_1" in merged_prov["merged_chunk_ids"]
    assert "c_workloads_2" in merged_prov["merged_chunk_ids"]
    assert "doc_a.pdf" in merged_prov["merged_sources"]
    assert "doc_b.pdf" in merged_prov["merged_sources"]


# ---------------------------------------------------------------------------
# Test J: Retention of Distinct Passages on Same Topic
# ---------------------------------------------------------------------------
def test_regression_retention_of_distinct_passages_on_same_topic():
    c1 = EvidenceChunk(
        chunk_id="c_resourcemanager",
        doc_id="uber.pdf",
        source="uber.pdf",
        text="The ResourceManager is the central master authority arbitrating cluster compute and memory allocation across distributed nodes.",
        similarity_score=0.80,
        rank=1
    )
    c2 = EvidenceChunk(
        chunk_id="c_capacity_scheduler",
        doc_id="uber.pdf",
        source="uber.pdf",
        text="The Capacity Scheduler coordinates multi-tenant hierarchical queues to guarantee minimum capacity and avoid starvation.",
        similarity_score=0.78,
        rank=2
    )
    col = EvidenceCollection(query="how does yarn work", chunks=[c1, c2], retriever_name="Hopfield")
    dedup_col, diagnostics = EvidenceDeduplicator.deduplicate(col)

    # Distinct passages must NOT be collapsed
    assert diagnostics["duplicates_removed"] == 0
    assert len(dedup_col.chunks) == 2


# ---------------------------------------------------------------------------
# Test K: Non-Repetition Enforcement in Synthesized Output
# ---------------------------------------------------------------------------
def test_regression_non_repetition_in_synthesis(yarn_evidence_collection):
    synthesizer = GenerativeRAGSynthesizer()
    answer = synthesizer.synthesize("where is yarn used", yarn_evidence_collection, style="conversational")

    # Count occurrences of key workload entities in the answer text
    text_lower = answer.answer.lower()
    mapreduce_count = text_lower.count("mapreduce")
    spark_count = text_lower.count("spark")
    flink_count = text_lower.count("flink")

    # Each concept must appear only once, not repeated across paragraphs
    assert mapreduce_count == 1, f"Expected 1 MapReduce mention, got {mapreduce_count}"
    assert spark_count == 1, f"Expected 1 Spark mention, got {spark_count}"
    assert flink_count == 1, f"Expected 1 Flink mention, got {flink_count}"


# ---------------------------------------------------------------------------
# Test L: Paraphrase Verification via Morphological Stemming
# ---------------------------------------------------------------------------
def test_regression_paraphrase_verification():
    chunks = [
        EvidenceChunk(
            chunk_id="chunk_arch",
            doc_id="uber.pdf",
            source="uber.pdf",
            text="YARN achieves distributed resource management through the coordination of three primary components: the ResourceManager, NodeManagers, and ApplicationMasters.",
            similarity_score=0.85,
            rank=1
        )
    ]
    # Paraphrase using derivational variants: "manages" -> "management", "allocates" -> "coordination"
    paraphrased_claim = "YARN manages distributed cluster resources through the coordination of NodeManagers and the ResourceManager [E1]."
    claims, _, _ = CitationValidator.extract_and_validate(paraphrased_claim, chunks)

    assert len(claims) == 1
    assert claims[0].is_verified is True
    assert claims[0].is_supported is True


# ---------------------------------------------------------------------------
# Test M: Hopfield Diagnostics & Trajectory Preservation
# ---------------------------------------------------------------------------
def test_regression_hopfield_diagnostics_preservation():
    c1 = EvidenceChunk(
        chunk_id="c1",
        doc_id="d1",
        source="d1.pdf",
        text="YARN functions as the cluster operating system.",
        similarity_score=0.9,
        rank=1
    )
    col = EvidenceCollection(
        query="what is YARN",
        chunks=[c1],
        retriever_name="Hopfield Associative Memory",
        attractor_energy=-12.345,
        energy_depth=4.56,
        is_grounded=True,
        trajectory_hops=[{"hop": 1, "energy": -12.345, "delta": 0.01}]
    )
    dedup_col, _ = EvidenceDeduplicator.deduplicate(col)

    # Hopfield energy landscape and trajectory must be preserved intact
    assert dedup_col.retriever_name == "Hopfield Associative Memory"
    assert dedup_col.attractor_energy == -12.345
    assert dedup_col.energy_depth == 4.56
    assert len(dedup_col.trajectory_hops) == 1


# ---------------------------------------------------------------------------
# Fixture: Cybersecurity Evidence Chunks
# ---------------------------------------------------------------------------
@pytest.fixture
def cybersecurity_evidence_collection() -> EvidenceCollection:
    """Fixture providing realistic chunks reflecting cyber security unit 2.pdf."""
    chunks = [
        # Chunk 1: Outline section on Security Metrics (simulating cyber security unit 2.pdf)
        EvidenceChunk(
            chunk_id="chunk_sec_metrics",
            doc_id="cyber_security_unit_2.pdf",
            source="cyber security unit 2.pdf",
            text=(
                "12. Security Metrics and Their Importance Systems Security metrics are critical tools "
                "for quantifying the effectiveness of security processes, measuring risk posture, and "
                "demonstrating compliance with standards. Metrics provide quantifiable data on system health."
            ),
            similarity_score=0.88,
            rank=1,
            page_number=12,
            is_core_attractor=True
        ),
        # Chunk 2: Security Logs and operational practices
        EvidenceChunk(
            chunk_id="chunk_sec_logs",
            doc_id="cyber_security_unit_2.pdf",
            source="cyber security unit 2.pdf",
            text=(
                "10. Security Logs and Their Role in Incident Response Systems Security logs are records of events "
                "occurring within an organization’s systems and networks. They play a vital role in: Threat detection: "
                "Identifying suspicious activity and potential breaches. Incident response: Providing the necessary information "
                "to investigate and mitigate security incidents. Forensic investigation: Supporting post-incident analysis "
                "to understand how a breach occurred."
            ),
            similarity_score=0.86,
            rank=2,
            page_number=10,
            is_core_attractor=False
        ),
        # Chunk 3: Security monitoring
        EvidenceChunk(
            chunk_id="chunk_sec_monitoring",
            doc_id="cyber_security_unit_2.pdf",
            source="cyber security unit 2.pdf",
            text=(
                "Security monitoring and event analysis enable real-time detection of unauthorized access, malware propagation, "
                "and vulnerability exploitation across enterprise systems."
            ),
            similarity_score=0.82,
            rank=3,
            page_number=11,
            is_core_attractor=False
        ),
    ]
    return EvidenceCollection(
        query="what is cybersecurity",
        chunks=chunks,
        retriever_name="HopfieldRetrieverAdapter"
    )


# ---------------------------------------------------------------------------
# Test N: Cybersecurity Definition Fidelity & Heading Separation
# ---------------------------------------------------------------------------
def test_regression_cybersecurity_definition_fidelity(cybersecurity_evidence_collection):
    synthesizer = GenerativeRAGSynthesizer()
    answer = synthesizer.synthesize("what is cybersecurity", cybersecurity_evidence_collection, style="conversational")

    assert answer.abstained is False
    assert answer.confidence > 0.50

    answer_text = answer.answer
    answer_lower = answer_text.lower()

    # 1. Opening sentence must NOT lead with "12. Security Metrics" or define cybersecurity as metrics
    assert not answer_text.startswith("12.")
    assert not answer_text.startswith("10.")
    assert "12. security metrics" not in answer_lower
    assert not answer_lower.startswith("security metrics are")

    # 2. Must directly state definition / operational scope first
    assert "operational practices" in answer_lower or "defined" in answer_lower or "security monitoring" in answer_lower
    assert any(term in answer_lower for term in ["threat detection", "incident response", "forensic investigation", "monitoring"])

    # 3. Honest attribution: acknowledging absence of a standalone textbook definition when relying on operational scope
    assert "within the provided documentation" in answer_lower or "operational practices" in answer_lower

    # 4. Valid citations present
    assert any(f"[E{i}]" in answer_text for i in [1, 2, 3])
    assert len(answer.claims) > 0
    assert all(c.is_verified for c in answer.claims)

    # 5. Verify executive and concise styles also maintain definition fidelity
    exec_answer = synthesizer.synthesize("what is cybersecurity", cybersecurity_evidence_collection, style="executive")
    assert not exec_answer.answer.startswith("12.")
    assert "operational practices" in exec_answer.answer.lower() or "threat detection" in exec_answer.answer.lower()

    concise_answer = synthesizer.synthesize("what is cybersecurity", cybersecurity_evidence_collection, style="concise")
    assert not concise_answer.answer.startswith("12.")
    assert len(concise_answer.answer.split(".")) <= 4


# ---------------------------------------------------------------------------
# Test O: Multi-Part Synthesis ("where is yarn used, and how does its architecture work")
# ---------------------------------------------------------------------------
def test_regression_yarn_multipart_synthesis(yarn_evidence_collection):
    synthesizer = GenerativeRAGSynthesizer()
    query = "where is yarn used, and how does its architecture work?"
    answer = synthesizer.synthesize(query, yarn_evidence_collection, style="conversational")

    assert answer.abstained is False
    assert answer.confidence > 0.60
    answer_lower = answer.answer.lower()

    # 1. Part 1 (Where): addresses environments and workloads
    assert "yarn is used in" in answer_lower or "used" in answer_lower
    assert any(w in answer_lower for w in ["multi-tenant", "workloads", "batch", "spark", "mapreduce"])

    # 2. Part 2 (How / Architecture): addresses components coordination
    assert "architecture" in answer_lower or "components" in answer_lower or "resourcemanager" in answer_lower
    assert "nodemanager" in answer_lower or "applicationmaster" in answer_lower

    # 3. Valid citations
    assert len(answer.claims) >= 2
    assert all(c.is_verified for c in answer.claims)


# ---------------------------------------------------------------------------
# Test P: Components and Responsibilities Breakdown
# ---------------------------------------------------------------------------
def test_regression_yarn_components_synthesis(yarn_evidence_collection):
    synthesizer = GenerativeRAGSynthesizer()
    query = "explain YARN components and their responsibilities"
    answer = synthesizer.synthesize(query, yarn_evidence_collection, style="conversational")

    assert answer.abstained is False
    assert answer.confidence > 0.60
    answer_lower = answer.answer.lower()

    # Must explicitly articulate the three primary components
    assert "resourcemanager" in answer_lower
    assert "nodemanager" in answer_lower
    assert "applicationmaster" in answer_lower

    # Must mention coordination / responsibilities
    assert "resources" in answer_lower or "containers" in answer_lower or "arbitrates" in answer_lower

    # Valid citations
    assert len(answer.claims) >= 1
    assert all(c.is_verified for c in answer.claims)

