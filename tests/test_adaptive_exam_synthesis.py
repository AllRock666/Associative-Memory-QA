"""
Tests for Adaptive Academic Exam Synthesis and Intent Classification
=====================================================================
Covers:
1. 2-mark examination answer (concise definition + essential technical fact).
2. 5-mark examination answer (definition + 3-4 structural components).
3. 8-mark examination answer (5-part academic outline with daemons, workflow, guarantees).
4. 10-mark examination answer (exhaustive breakdown + markdown comparative matrix table).
5. Comparison genre (markdown dimension table).
6. Worked example genre (scenario setup, step execution, verification state).
7. Structured notes genre (key terms, acronyms, component cheat sheet, high-yield takeaways).
8. FormatDetector classification accuracy across diverse query phrasings.
"""

import pytest
from core.format_detector import FormatDetector
from core.format_guidance import FORMAT_CATALOG
from core.schemas import EvidenceChunk, EvidenceCollection
from core.generative_synthesizer import GenerativeRAGSynthesizer
from core.llm_provider import MockLLMProvider


@pytest.fixture
def yarn_evidence():
    chunks = [
        EvidenceChunk(
            chunk_id="c1",
            doc_id="yarn.pdf",
            source="yarn.pdf",
            text="Apache Hadoop YARN (Yet Another Resource Negotiator) is the architectural redesign of Hadoop that decouples resource allocation from job scheduling.",
            similarity_score=0.92,
            is_core_attractor=True
        ),
        EvidenceChunk(
            chunk_id="c2",
            doc_id="yarn.pdf",
            source="yarn.pdf",
            text="The ResourceManager functions as the cluster master daemon comprising the Scheduler and ApplicationsManager.",
            similarity_score=0.88,
            is_core_attractor=True
        ),
        EvidenceChunk(
            chunk_id="c3",
            doc_id="yarn.pdf",
            source="yarn.pdf",
            text="NodeManagers run on each cluster node to launch and monitor resource containers consisting of allocated memory and CPU cores.",
            similarity_score=0.85,
            is_core_attractor=True
        ),
        EvidenceChunk(
            chunk_id="c4",
            doc_id="yarn.pdf",
            source="yarn.pdf",
            text="The ApplicationMaster is instantiated per application to negotiate resource containers dynamically and coordinate task execution.",
            similarity_score=0.83,
            is_core_attractor=True
        ),
        EvidenceChunk(
            chunk_id="c5",
            doc_id="yarn.pdf",
            source="yarn.pdf",
            text="YARN enables multi-tenancy allowing MapReduce, Apache Spark, and Apache Flink to share the same physical cluster hardware.",
            similarity_score=0.80,
            is_core_attractor=True
        ),
    ]
    return EvidenceCollection(query="Explain YARN", chunks=chunks, retriever_name="Hopfield Memory")


# --- Test 1: FormatDetector Classification ---
def test_format_detector_marks_and_genres():
    # 2 marks
    fid, _ = FormatDetector.detect_format("What is cybersecurity? (2-mark answer)")
    assert fid == "2_mark"

    # 5 marks
    fid, _ = FormatDetector.detect_format("Explain YARN architecture for 5 marks")
    assert fid == "5_mark"

    # 8 marks
    fid, _ = FormatDetector.detect_format("Discuss YARN components as an 8-mark question")
    assert fid == "8_mark"

    # 10 marks
    fid, _ = FormatDetector.detect_format("Provide a 10-mark comprehensive answer on YARN internals")
    assert fid == "10_mark"

    # Comparison
    fid, _ = FormatDetector.detect_format("Compare MapReduce vs Spark on YARN")
    assert fid == "comparison"

    # Worked Example
    fid, _ = FormatDetector.detect_format("Give a worked example of job submission in YARN")
    assert fid == "worked_example"

    # Revision Notes
    fid, _ = FormatDetector.detect_format("Create revision notes and cheat sheet for YARN daemons")
    assert fid == "structured_notes"

    # Explicit preference override
    fid, _ = FormatDetector.detect_format("Tell me about YARN", explicit_preference="8_mark")
    assert fid == "8_mark"


# --- Test 2: 2-Mark Answer Synthesis ---
def test_2_mark_synthesis(yarn_evidence):
    synth = GenerativeRAGSynthesizer(llm_provider=MockLLMProvider())
    ans = synth.synthesize(query="What is YARN for 2 marks?", evidence=yarn_evidence, style="2_mark")

    assert not ans.abstained
    words = ans.answer.split()
    assert 25 <= len(words) <= 120
    assert "YARN" in ans.answer
    assert "[E" in ans.answer


# --- Test 3: 5-Mark Answer Synthesis ---
def test_5_mark_synthesis(yarn_evidence):
    synth = GenerativeRAGSynthesizer(llm_provider=MockLLMProvider())
    ans = synth.synthesize(query="Explain YARN for 5 marks", evidence=yarn_evidence, style="5_mark")

    assert not ans.abstained
    words = ans.answer.split()
    assert len(words) >= 70
    assert "### Core Architectural Definition" in ans.answer
    assert "### Key Daemons & Operational Responsibilities" in ans.answer


# --- Test 4: 8-Mark Answer Synthesis ---
def test_8_mark_synthesis(yarn_evidence):
    synth = GenerativeRAGSynthesizer(llm_provider=MockLLMProvider())
    ans = synth.synthesize(query="Explain YARN architecture as an 8-mark question", evidence=yarn_evidence, style="8_mark")

    assert not ans.abstained
    words = ans.answer.split()
    assert len(words) >= 140
    assert "### 1. Architectural Foundation & Decoupling Principle" in ans.answer
    assert "### 2. Core Daemons & Component Responsibilities" in ans.answer
    assert "### 3. Application Lifecycle & Container Negotiation Workflow" in ans.answer
    assert "### 4. Multi-Tenancy & Workload Execution Capabilities" in ans.answer
    assert "### 5. Architectural Trade-offs & Operational Guarantees" in ans.answer


# --- Test 5: 10-Mark Answer Synthesis with Comparative Matrix Table ---
def test_10_mark_synthesis_with_matrix_table(yarn_evidence):
    synth = GenerativeRAGSynthesizer(llm_provider=MockLLMProvider())
    ans = synth.synthesize(query="Explain YARN in depth for 10 marks", evidence=yarn_evidence, style="10_mark")

    assert not ans.abstained
    words = ans.answer.split()
    assert len(words) >= 180
    assert "### 1. Comprehensive Architectural Foundation & Problem Context" in ans.answer
    assert "### 2. Component Daemons & Structural Breakdown" in ans.answer
    assert "### 4. Comparative Architectural Paradigm Matrix" in ans.answer
    # Must include markdown table
    assert "| Architectural Dimension | Primary Paradigm | Alternative / Complementary Paradigm | Citations |" in ans.answer
    assert "| Resource Allocation |" in ans.answer
    assert "### 5. Real-World Multi-Tenant Workloads & Framework Diversity" in ans.answer


# --- Test 6: Comparison Table Genre ---
def test_comparison_genre_synthesis(yarn_evidence):
    synth = GenerativeRAGSynthesizer(llm_provider=MockLLMProvider())
    ans = synth.synthesize(query="Compare YARN vs classic MapReduce", evidence=yarn_evidence, style="comparison")

    assert not ans.abstained
    assert "### Dimension-by-Dimension Comparison" in ans.answer
    assert "| Architectural Dimension |" in ans.answer
    assert "| Operational Scope |" in ans.answer


# --- Test 7: Worked Example Genre ---
def test_worked_example_synthesis(yarn_evidence):
    synth = GenerativeRAGSynthesizer(llm_provider=MockLLMProvider())
    ans = synth.synthesize(query="Provide a worked example of YARN job execution", evidence=yarn_evidence, style="worked_example")

    assert not ans.abstained
    assert "### 1. Concrete Scenario Setup" in ans.answer
    assert "### 2. Step-by-Step Execution Walkthrough" in ans.answer
    assert "**Step 1**:" in ans.answer


# --- Test 8: Structured Revision Notes Genre ---
def test_structured_notes_synthesis(yarn_evidence):
    synth = GenerativeRAGSynthesizer(llm_provider=MockLLMProvider())
    ans = synth.synthesize(query="Create revision notes for YARN", evidence=yarn_evidence, style="structured_notes")

    assert not ans.abstained
    assert "### Key Definitions & Acronyms" in ans.answer
    assert "### Architecture & Component Cheat-Sheet" in ans.answer
    assert "### High-Yield Exam Takeaways" in ans.answer
