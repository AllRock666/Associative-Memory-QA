"""
Test Suite: Answer Quality, Adaptive Depth, Grounding, and Formatting
=====================================================================
Validates the complete production question-answering pipeline:
1. Simple factual questions receive direct, compact answers (2-4 sentences).
2. Conceptual definitions lead with the definition first (no sub-concept confusion).
3. Multi-component technical questions cover all components with substantive depth.
4. Multi-part questions address every question aspect explicitly.
5. Comparisons produce clear dimensional contrasts and Markdown summary tables.
6. Mechanism/how questions explain workflows and component coordination.
7. Insufficient evidence queries trigger honest abstention with zero hallucination.
8. Cross-topic distractor passages are rejected from the synthesis.
9. Complex queries produce substantive answers with verified claims.
10. Provider truncation and unclosed JSON strings are gracefully recovered.
11. All four answer styles (detailed, conversational, executive, concise) adapt correctly.
12. Markdown formatting (headers, tables, numbered lists, citations) renders properly.
"""

import pytest
import re
from core.schemas import EvidenceChunk, EvidenceCollection, GroundedAnswer
from core.llm_provider import (
    LLMConfig,
    MockLLMProvider,
    build_grounded_prompt,
    parse_structured_llm_output,
)
from core.generative_synthesizer import (
    GenerativeRAGSynthesizer,
    CitationValidator,
    AbstentionEngine,
    classify_intent
)


@pytest.fixture
def rich_yarn_evidence() -> EvidenceCollection:
    """Fixture providing comprehensive YARN evidence chunks."""
    chunks = [
        EvidenceChunk(
            chunk_id="c_yarn_def",
            doc_id="uber_case_study.pdf",
            source="Uber Big Data Case Study.pdf",
            text=(
                "Yet Another Resource Negotiator (YARN) functions as the cluster operating system, "
                "fundamentally decoupling resource management and job scheduling from the underlying "
                "data processing engines."
            ),
            similarity_score=0.92,
            rank=1,
            page_number=4,
            is_core_attractor=True
        ),
        EvidenceChunk(
            chunk_id="c_yarn_workloads",
            doc_id="uber_case_study.pdf",
            source="Uber Big Data Case Study.pdf",
            text=(
                "In a vast, multi-tenant engineering environment like Uber's, YARN arbitrates resources across "
                "diverse data workloads, ranging from massive, resource-intensive batch MapReduce ETL jobs "
                "to interactive, low-latency Spark SQL queries and streaming Flink applications. "
                "YARN is the critical middleware enabling heterogeneous workloads to coexist efficiently."
            ),
            similarity_score=0.88,
            rank=2,
            page_number=5,
            is_core_attractor=True
        ),
        EvidenceChunk(
            chunk_id="c_yarn_arch",
            doc_id="uber_case_study.pdf",
            source="Uber Big Data Case Study.pdf",
            text=(
                "YARN achieves distributed resource management through the coordination of three primary "
                "components: the ResourceManager (RM), NodeManagers (NM), and ApplicationMasters (AM). "
                "The centralized ResourceManager arbitrates global resources across the entire cluster."
            ),
            similarity_score=0.85,
            rank=3,
            page_number=5,
            is_core_attractor=False
        ),
        EvidenceChunk(
            chunk_id="c_yarn_nm",
            doc_id="uber_case_study.pdf",
            source="Uber Big Data Case Study.pdf",
            text=(
                "NodeManagers function as per-node daemons that launch, monitor, and enforce resource units "
                "known as containers on individual machines. NodeManagers send periodic heartbeats to the ResourceManager."
            ),
            similarity_score=0.81,
            rank=4,
            page_number=6,
            is_core_attractor=False
        ),
        EvidenceChunk(
            chunk_id="c_yarn_am",
            doc_id="uber_case_study.pdf",
            source="Uber Big Data Case Study.pdf",
            text=(
                "The ApplicationMaster is an application-specific entity responsible for negotiating appropriate "
                "resource containers from the ResourceManager and tracking the execution lifecycle of tasks."
            ),
            similarity_score=0.79,
            rank=5,
            page_number=6,
            is_core_attractor=False
        ),
        EvidenceChunk(
            chunk_id="c_yarn_guarantees",
            doc_id="uber_case_study.pdf",
            source="Uber Big Data Case Study.pdf",
            text=(
                "By decoupling resource negotiation from processing logic, YARN guarantees multi-tenant isolation, "
                "prevents resource starvation, eliminates cluster bottlenecks, and ensures robust fault tolerance."
            ),
            similarity_score=0.75,
            rank=6,
            page_number=7,
            is_core_attractor=False
        ),
        EvidenceChunk(
            chunk_id="c_hdfs_storage",
            doc_id="uber_case_study.pdf",
            source="Uber Big Data Case Study.pdf",
            text=(
                "Hadoop Distributed File System (HDFS) provides scalable distributed storage across the cluster, "
                "dividing large files into fixed-size 128MB blocks replicated across DataNodes managed by a NameNode."
            ),
            similarity_score=0.60,
            rank=7,
            page_number=3,
            is_core_attractor=False
        ),
    ]
    return EvidenceCollection(query="Explain YARN components", chunks=chunks, retriever_name="HopfieldRetriever")


@pytest.fixture
def cybersecurity_evidence() -> EvidenceCollection:
    """Fixture providing cybersecurity and security telemetry evidence chunks."""
    chunks = [
        EvidenceChunk(
            chunk_id="c_sec_1",
            doc_id="security_telemetry.pdf",
            source="Enterprise Security Telemetry.pdf",
            text=(
                "Security logs play a vital role in identifying suspicious activities, detecting unauthorized access "
                "attempts, and investigating security incidents and potential breaches."
            ),
            similarity_score=0.89,
            rank=1,
            page_number=2,
            is_core_attractor=True
        ),
        EvidenceChunk(
            chunk_id="c_sec_2",
            doc_id="security_telemetry.pdf",
            source="Enterprise Security Telemetry.pdf",
            text=(
                "Security logs provide an essential foundation for threat detection, incident response, and "
                "forensic investigation by recording authentication attempts and network traffic patterns."
            ),
            similarity_score=0.85,
            rank=2,
            page_number=3,
            is_core_attractor=True
        ),
        EvidenceChunk(
            chunk_id="c_sec_3",
            doc_id="security_telemetry.pdf",
            source="Enterprise Security Telemetry.pdf",
            text=(
                "Security metrics are measurable values used by security teams to assess the performance of cybersecurity "
                "controls, including Mean Time to Detect (MTTD) and Mean Time to Respond (MTTR)."
            ),
            similarity_score=0.82,
            rank=3,
            page_number=8,
            is_core_attractor=False
        ),
    ]
    return EvidenceCollection(query="What is cybersecurity?", chunks=chunks, retriever_name="HopfieldRetriever")


# ---------------------------------------------------------------------------
# Test 1: Simple factual question receives direct, compact answer
# ---------------------------------------------------------------------------
def test_simple_factual_depth(rich_yarn_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="What block size does HDFS use?",
        evidence=rich_yarn_evidence,
        style="concise"
    )

    assert not res.abstained
    assert "128MB" in res.answer or "128" in res.answer
    assert "[E" in res.answer
    # Should be concise (2-4 sentences, under 80 words)
    words = res.answer.split()
    assert len(words) < 80
    assert len(res.claims) >= 1


# ---------------------------------------------------------------------------
# Test 2: Conceptual definition leads with definition first
# ---------------------------------------------------------------------------
def test_conceptual_definition_depth(cybersecurity_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="What is cybersecurity?",
        evidence=cybersecurity_evidence,
        style="detailed"
    )

    assert not res.abstained
    # Must NOT define cybersecurity as metrics or logs
    first_sentence = res.answer.split('.')[0]
    assert "metrics are measurable values" not in first_sentence.lower()
    # Must state definition through operational practices
    assert "cybersecurity" in first_sentence.lower()
    assert any(term in res.answer.lower() for term in ["threat detection", "incident response", "security logs"])
    # Detailed style should have multiple sections
    assert "### " in res.answer
    assert len(res.claims) >= 3
    assert all(c.is_verified for c in res.claims)


# ---------------------------------------------------------------------------
# Test 3: Multi-component technical question provides comprehensive depth
# ---------------------------------------------------------------------------
def test_multi_component_architecture_depth(rich_yarn_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="Explain YARN components and their responsibilities.",
        evidence=rich_yarn_evidence,
        style="detailed"
    )

    assert not res.abstained
    ans_lower = res.answer.lower()

    # Opening must directly answer the primary intent
    first_sentence = res.answer.split('.')[0]
    assert "resourcemanager" in first_sentence.lower() or "components" in first_sentence.lower()

    # Must cover all core components and daemons
    assert "resourcemanager" in ans_lower
    assert "nodemanager" in ans_lower
    assert "applicationmaster" in ans_lower
    assert "container" in ans_lower

    # Substantive length and multiple structured sections
    assert "### Core Daemons & Component Responsibilities" in res.answer
    words = res.answer.split()
    assert len(words) >= 120

    # High citation verification rate
    assert len(res.claims) >= 5
    verified_rate = sum(1 for c in res.claims if c.is_verified) / len(res.claims)
    assert verified_rate >= 0.85


# ---------------------------------------------------------------------------
# Test 4: Multi-part questions address every aspect explicitly
# ---------------------------------------------------------------------------
def test_multipart_question_depth(rich_yarn_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="Where is YARN used, and how does its architecture work?",
        evidence=rich_yarn_evidence,
        style="detailed"
    )

    assert not res.abstained
    ans_lower = res.answer.lower()

    # Part 1 (Where used)
    assert any(w in ans_lower for w in ["multi-tenant", "cluster", "workload", "distributed"])
    # Part 2 (Architecture)
    assert any(w in ans_lower for w in ["decoupling", "resourcemanager", "nodemanager", "components"])

    # First section must address where it is used
    first_p = res.answer.split('\n\n')[0]
    assert "used in" in first_p.lower() or "used" in first_p.lower()


# ---------------------------------------------------------------------------
# Test 5: Comparative analysis produces contrast and summary matrix
# ---------------------------------------------------------------------------
def test_comparison_depth_and_matrix(rich_yarn_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="What is the difference between HDFS and YARN?",
        evidence=rich_yarn_evidence,
        style="detailed"
    )

    assert not res.abstained
    # Must contrast compute vs storage
    assert "difference" in res.answer.lower() or "contrast" in res.answer.lower()
    # Must contain Markdown comparison table
    assert "### Comparative Summary Matrix" in res.answer
    assert "| Architectural Dimension |" in res.answer
    assert "| Primary Role |" in res.answer


# ---------------------------------------------------------------------------
# Test 6: Mechanism/how question explains workflow and component coordination
# ---------------------------------------------------------------------------
def test_how_mechanism_depth(rich_yarn_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="How does YARN manage cluster resources?",
        evidence=rich_yarn_evidence,
        style="conversational"
    )

    assert not res.abstained
    ans_lower = res.answer.lower()
    # Direct answer first without filler preamble
    assert not res.answer.startswith("Based on the provided documents")
    assert "resourcemanager" in ans_lower or "coordination" in ans_lower
    # Must cite evidence
    assert "[E" in res.answer


# ---------------------------------------------------------------------------
# Test 7: Insufficient evidence triggers honest abstention
# ---------------------------------------------------------------------------
def test_insufficient_evidence_abstention(rich_yarn_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="What is quantum error correction in topological qubits?",
        evidence=rich_yarn_evidence,
        style="detailed"
    )

    assert res.abstained is True
    assert res.confidence == 0.0
    assert "cannot answer" in res.answer.lower() or "insufficient" in res.answer.lower()


# ---------------------------------------------------------------------------
# Test 8: Cross-topic distractor rejection
# ---------------------------------------------------------------------------
def test_cross_topic_rejection(rich_yarn_evidence, cybersecurity_evidence):
    # Combine YARN chunks with a Cybersecurity distractor chunk
    mixed_chunks = list(rich_yarn_evidence.chunks) + list(cybersecurity_evidence.chunks)
    mixed_evidence = EvidenceCollection(
        query="Explain YARN components and their responsibilities",
        chunks=mixed_chunks,
        retriever_name="HopfieldRetriever"
    )

    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="Explain YARN components and their responsibilities",
        evidence=mixed_evidence,
        style="detailed"
    )

    assert not res.abstained
    # The answer about YARN should NOT mention security logs or MTTD
    assert "security logs" not in res.answer.lower()
    assert "mttd" not in res.answer.lower()
    assert "resourcemanager" in res.answer.lower()


# ---------------------------------------------------------------------------
# Test 9: Complex query produces substantive answer with high verified ratio
# ---------------------------------------------------------------------------
def test_long_complex_answer_substance(rich_yarn_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="Provide a comprehensive breakdown of YARN architecture, daemons, supported workloads, and isolation guarantees.",
        evidence=rich_yarn_evidence,
        style="detailed"
    )

    assert not res.abstained
    words = res.answer.split()
    assert len(words) >= 140
    # Must contain multiple ### subsections
    sections = re.findall(r'### [^\n]+', res.answer)
    assert len(sections) >= 2
    # Verified claims
    assert len(res.claims) >= 6
    assert sum(1 for c in res.claims if c.is_verified) >= 5


# ---------------------------------------------------------------------------
# Test 10: Provider truncation and unclosed JSON recovery
# ---------------------------------------------------------------------------
def test_unclosed_json_recovery():
    truncated_llm_output = (
        '{"answer": "YARN arbitrates cluster resources across execution frameworks [E1]. '
        'NodeManagers monitor containers on individual machines [E2].", '
        '"claims": [{"text": "YARN arbitrates cluster resources", "evidence_ids": ["E1"]'
        # Missing closing brackets and braces!
    )

    ans, claims, is_abstained, reason = parse_structured_llm_output(truncated_llm_output)

    assert is_abstained is False
    assert "YARN arbitrates cluster resources" in ans
    assert "NodeManagers monitor containers" in ans
    assert len(claims) >= 2
    assert any("E1" in c["evidence_ids"] for c in claims)


# ---------------------------------------------------------------------------
# Test 11: All four answer styles adapt correctly
# ---------------------------------------------------------------------------
def test_all_four_answer_styles(rich_yarn_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    query = "Explain YARN components and their responsibilities."

    # 1. Detailed: headers, extensive sections
    res_det = synthesizer.synthesize(query=query, evidence=rich_yarn_evidence, style="detailed")
    assert "### " in res_det.answer
    assert len(res_det.answer.split()) >= 100

    # 2. Conversational: narrative paragraphs
    res_conv = synthesizer.synthesize(query=query, evidence=rich_yarn_evidence, style="conversational")
    assert not res_conv.answer.startswith("###")
    assert "\n\n" in res_conv.answer

    # 3. Executive: structured summary with bullet points
    res_exec = synthesizer.synthesize(query=query, evidence=rich_yarn_evidence, style="executive")
    assert "### Executive Summary" in res_exec.answer
    assert "- **Key Finding**:" in res_exec.answer
    assert "- **Core Architecture & Entities**:" in res_exec.answer

    # 4. Concise: short and direct
    res_conc = synthesizer.synthesize(query=query, evidence=rich_yarn_evidence, style="concise")
    assert "###" not in res_conc.answer
    assert len(res_conc.answer.split()) < 60
    assert len(res_conc.answer.split()) < len(res_det.answer.split())


# ---------------------------------------------------------------------------
# Test 12: Markdown tables, lists, and citations syntax
# ---------------------------------------------------------------------------
def test_markdown_tables_and_lists_syntax(rich_yarn_evidence):
    synthesizer = GenerativeRAGSynthesizer()
    res = synthesizer.synthesize(
        query="What is the difference between HDFS and YARN?",
        evidence=rich_yarn_evidence,
        style="detailed"
    )

    # Check Markdown table syntax
    assert "| Architectural Dimension |" in res.answer
    assert "|---|" in res.answer or "| --- |" in res.answer
    table_lines = [line for line in res.answer.split('\n') if line.strip().startswith('|')]
    assert len(table_lines) >= 4  # Header, delimiter, and at least 2 data rows

    # Verify citation format [E#]
    citations = re.findall(r'\[E\d+\]', res.answer)
    assert len(citations) >= 2


# ---------------------------------------------------------------------------
# Test 13: Multi-Turn Coreference Resolution (Contextualized Querying)
# ---------------------------------------------------------------------------
def test_multi_turn_coreference_resolution():
    from core.query_handler import QueryProcessor

    processor = QueryProcessor()
    history = [
        {"role": "user", "content": "What is Apache YARN?"},
        {"role": "assistant", "content": "Apache YARN is a distributed cluster operating system and resource manager."}
    ]

    # 1. Anaphoric pronoun "it" resolved to "Apache YARN"
    q1 = "Where is it used?"
    res1 = processor.process(q1, conversation_history=history)
    assert "apache yarn" in res1.cleaned.lower()
    assert "where is apache yarn used" in res1.cleaned.lower()
    assert res1.metadata.get("contextualized_query") is not None

    # 2. Possessive pronoun "its" resolved
    q2 = "Explain its core components"
    res2 = processor.process(q2, conversation_history=history)
    assert "apache yarn" in res2.cleaned.lower()

    # 3. Direct query without anaphora left intact
    q3 = "What is cybersecurity?"
    res3 = processor.process(q3, conversation_history=history)
    assert "cybersecurity" in res3.cleaned.lower()
    assert "apache yarn" not in res3.cleaned.lower()


# ---------------------------------------------------------------------------
# Test 14: Compound Entity Expansion in Associative Retrieval
# ---------------------------------------------------------------------------
def test_compound_entity_expansion_associative():
    import torch
    from core.associative_engine import HopfieldAssociativeEngine
    from core.hopfield_torch import PyTorchHopfieldMemory

    memory = PyTorchHopfieldMemory()
    device = memory.device
    patterns = torch.randn(3, 64, device=device)
    patterns = torch.nn.functional.normalize(patterns, p=2, dim=-1)

    metadata = [
        {"chunk_id": "c1", "text": "Network security and cyber defense logs are vital for monitoring.", "source": "sec.pdf"},
        {"chunk_id": "c2", "text": "Multi-tenant clusters require strong isolation between jobs.", "source": "cluster.pdf"},
        {"chunk_id": "c3", "text": "Microservices communicate via lightweight HTTP or gRPC interfaces.", "source": "arch.pdf"},
    ]
    memory.store(patterns.cpu().numpy(), metadata)
    engine = HopfieldAssociativeEngine(memory=memory)

    # Query with compound word "cybersecurity"
    res = engine.retrieve_associative(
        query_embedding=patterns[0].cpu().numpy(),
        top_k=2,
        raw_query="What is cybersecurity?"
    )

    # c1 should be activated as core attractor basin
    top_ids = [c.chunk_id for c in res.core_attractor_chunks]
    assert "c1" in top_ids


# ---------------------------------------------------------------------------
# Test 15: General Technical Domain Dynamic Deep Synthesis
# ---------------------------------------------------------------------------
def test_general_domain_dynamic_deep_synthesis():
    chunks = [
        EvidenceChunk(
            chunk_id="c_trans_1",
            doc_id="transformer_paper.pdf",
            source="Attention Is All You Need.pdf",
            text=(
                "The Transformer architecture relies entirely on self-attention mechanisms to compute "
                "representations of its input and output without using sequence-aligned recurrent networks."
            ),
            similarity_score=0.91,
            rank=1,
            page_number=1,
            is_core_attractor=True
        ),
        EvidenceChunk(
            chunk_id="c_trans_2",
            doc_id="transformer_paper.pdf",
            source="Attention Is All You Need.pdf",
            text=(
                "Multi-head attention allows the model to jointly attend to information from different "
                "representation subspaces at different positions across the sequence."
            ),
            similarity_score=0.87,
            rank=2,
            page_number=3,
            is_core_attractor=True
        ),
        EvidenceChunk(
            chunk_id="c_trans_3",
            doc_id="transformer_paper.pdf",
            source="Attention Is All You Need.pdf",
            text=(
                "Scaled dot-product attention computes compatibility scores between queries and keys, "
                "scaling by the square root of key dimension to prevent vanishing gradients."
            ),
            similarity_score=0.84,
            rank=3,
            page_number=4,
            is_core_attractor=False
        ),
        EvidenceChunk(
            chunk_id="c_trans_4",
            doc_id="transformer_paper.pdf",
            source="Attention Is All You Need.pdf",
            text=(
                "By dispensing with recurrence, the Transformer achieves significant training speedups, "
                "enhanced scalability, and superior parallelization efficiency on modern GPU accelerators."
            ),
            similarity_score=0.80,
            rank=4,
            page_number=5,
            is_core_attractor=False
        ),
    ]
    evidence = EvidenceCollection(query="Explain the Transformer architecture", chunks=chunks, retriever_name="HopfieldRetriever")
    synthesizer = GenerativeRAGSynthesizer(llm_config=LLMConfig(provider="deterministic"))

    # Detailed style on general domain
    res = synthesizer.synthesize(
        query="Explain the Transformer architecture and its attention mechanisms.",
        evidence=evidence,
        style="detailed"
    )

    # 1. Direct answer first
    assert "Transformer architecture" in res.answer
    assert "[E1]" in res.answer

    # 2. Dynamic thematic sections generated for general domain
    assert "### Structural Foundations & Core Principles" in res.answer
    assert "### Operational Mechanics & Process Workflow" in res.answer
    assert "### Performance Characteristics & Technical Guarantees" in res.answer

    # 3. Substantive content
    assert "Multi-head attention" in res.answer
    assert "Scaled dot-product attention" in res.answer


# ---------------------------------------------------------------------------
# Test 16: Sentence and Artifact Smoothing
# ---------------------------------------------------------------------------
def test_sentence_and_artifact_smoothing():
    from core.generative_synthesizer import smooth_evidence_sentence

    # Slide / bullet prefix stripped
    assert smooth_evidence_sentence("• Deployment: Serve via FastAPI and Uvicorn.") == "Serve via FastAPI and Uvicorn."
    assert smooth_evidence_sentence("- Overview: The system architecture provides distributed memory.") == "The system architecture provides distributed memory."
    assert smooth_evidence_sentence("Slide 3: Scalable compute clusters and") == "Scalable compute clusters"
    assert smooth_evidence_sentence("· Key takeaways: Modern Hopfield Networks converge.") == "Modern Hopfield Networks converge."

