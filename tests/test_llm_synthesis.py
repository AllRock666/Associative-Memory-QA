"""
Test Suite: LLM-Powered Answer Generation Layer (Offline & Deterministic)
========================================================================
Validates:
1. MockLLMProvider normal generation with fluent grounded answers & [E1] citations.
2. Citation repair: stripping/handling hallucinated citation IDs like [E99].
3. Low-grounding / unsupported claims fallback handling.
4. Honest abstention propagation (INSUFFICIENT_EVIDENCE).
5. Answer styles (Conversational, Executive, Concise) prompt construction.
6. Timeout simulation triggering seamless fallback to deterministic engine.
7. Connection error simulation triggering deterministic fallback.
8. Truthful reporting: provider_used and fallback_occurred are never falsified.
9. Monotonic stage latency accounting (dedup_ms, synthesis_ms, validation_ms).
10. Parsing structured LLM outputs (JSON fence, raw JSON, plain text fallback).
11. OllamaProvider fast offline check (under 2.5s, cached).
12. FastAPI GET /api/status endpoint reporting LLM configuration.
13. FastAPI POST /api/search supporting dynamic llm_provider override.
14. HopfieldQAPipeline end-to-end query with mock LLM provider.
"""

import pytest
import time
from fastapi.testclient import TestClient

from core.schemas import EvidenceChunk, EvidenceCollection, GroundedAnswer
from core.llm_provider import (
    LLMConfig,
    MockLLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    build_grounded_prompt,
    parse_structured_llm_output,
    LLMTimeoutError,
    LLMConnectionError
)
from core.generative_synthesizer import GenerativeRAGSynthesizer, CitationValidator
from core.pipeline import HopfieldQAPipeline, PipelineConfig
from api import app, startup_event


@pytest.fixture
def sample_evidence() -> EvidenceCollection:
    """Fixture providing realistic evidence chunks for YARN questions."""
    return EvidenceCollection(
        query="where is yarn used",
        retriever_name="hopfield",
        chunks=[
            EvidenceChunk(
                chunk_id="chunk_yarn_1",
                doc_id="uber_case_study.pdf",
                source="Uber Big Data Case Study.pdf",
                text=(
                    "Yet Another Resource Negotiator (YARN) functions as the cluster operating system, "
                    "separating resource management from job scheduling and execution frameworks."
                ),
                similarity_score=0.88,
                rank=1,
                page_number=1,
                is_core_attractor=True
            ),
            EvidenceChunk(
                chunk_id="chunk_yarn_2",
                doc_id="uber_case_study.pdf",
                source="Uber Big Data Case Study.pdf",
                text=(
                    "YARN enables heterogeneous workloads across Hadoop clusters, scheduling diverse "
                    "engines including MapReduce batch processing, Spark SQL analytics, and Flink streaming."
                ),
                similarity_score=0.82,
                rank=2,
                page_number=2,
                is_core_attractor=False
            )
        ]
    )


# ---------------------------------------------------------------------------
# Test 1: Mock LLM normal generation with valid citations
# ---------------------------------------------------------------------------
def test_mock_llm_normal_generation(sample_evidence):
    config = LLMConfig(provider="mock", model="mock-model", fallback_to_deterministic=True)
    synthesizer = GenerativeRAGSynthesizer(llm_config=config)

    result = synthesizer.synthesize("where is yarn used", sample_evidence, style="conversational")

    assert isinstance(result, GroundedAnswer)
    assert result.abstained is False
    assert result.provider_used == "mock"
    assert result.model_name == "mock-model"
    assert result.fallback_occurred is False
    assert "[E1]" in result.answer or "[E2]" in result.answer
    assert len(result.claims) > 0
    assert any(c.is_verified for c in result.claims)
    assert result.confidence > 0.5


# ---------------------------------------------------------------------------
# Test 2: Citation repair removes/handles hallucinated citation IDs
# ---------------------------------------------------------------------------
def test_citation_repair_removes_hallucinated_citations():
    text_with_hallucinations = (
        "YARN manages cluster resources [E1]. "
        "It coordinates across multiple compute engines [E2]. "
        "It also communicates with external quantum processors [E99]."
    )

    chunks = [
        EvidenceChunk(chunk_id="c1", doc_id="d1", source="s1", text="YARN manages cluster resources.", similarity_score=0.9, rank=1),
        EvidenceChunk(chunk_id="c2", doc_id="d2", source="s2", text="It coordinates across compute engines.", similarity_score=0.8, rank=2),
    ]

    # repair_citations removes [E99] because valid_chunks has only 2 items
    repaired = CitationValidator.repair_citations(text_with_hallucinations, valid_chunks=chunks)
    assert "[E99]" not in repaired
    assert "[E1]" in repaired
    assert "[E2]" in repaired

    # extract_and_validate on repaired text
    claims, cited_chunks, unresolved = CitationValidator.extract_and_validate(repaired, chunks)
    assert len(unresolved) == 0
    assert len(cited_chunks) == 2


# ---------------------------------------------------------------------------
# Test 3: Low-grounding / unsupported claims fallback handling
# ---------------------------------------------------------------------------
def test_llm_unsupported_claims_triggers_fallback(sample_evidence):
    # Provider generates claims about unrelated topics (zero lexical overlap with YARN chunks)
    hallucinated_response = (
        '{"answer": "Mars has two moons named Phobos and Deimos [E1]. Atmospheric pressure is very low [E2].", '
        '"claims": ["Mars has two moons named Phobos and Deimos", "Atmospheric pressure is very low"]}'
    )
    provider = MockLLMProvider(custom_response=hallucinated_response)
    config = LLMConfig(provider="mock", fallback_to_deterministic=True)
    synthesizer = GenerativeRAGSynthesizer(llm_config=config, llm_provider=provider)

    result = synthesizer.synthesize("where is yarn used", sample_evidence)

    # Must fall back to deterministic engine because LLM hallucinated
    assert result.fallback_occurred is True
    assert "grounding" in (result.fallback_reason or "").lower() or "claims" in (result.fallback_reason or "").lower()
    assert result.provider_used == "deterministic_grounded_engine"
    # The deterministic fallback should contain YARN facts from sample_evidence
    assert "YARN" in result.answer


# ---------------------------------------------------------------------------
# Test 4: Honest abstention propagation
# ---------------------------------------------------------------------------
def test_mock_llm_insufficient_evidence_abstention(sample_evidence):
    # LLM explicitly abstains due to insufficient evidence
    provider = MockLLMProvider(custom_response="INSUFFICIENT_EVIDENCE")
    config = LLMConfig(provider="mock", fallback_to_deterministic=False)
    synthesizer = GenerativeRAGSynthesizer(llm_config=config, llm_provider=provider)

    result = synthesizer.synthesize("what is the capital of Atlantis", sample_evidence)

    assert result.abstained is True
    assert result.confidence == 0.0
    assert result.abstention_reason is not None


# ---------------------------------------------------------------------------
# Test 5: Answer styles prompt construction
# ---------------------------------------------------------------------------
def test_answer_styles_prompt_formatting(sample_evidence):
    # Conversational
    sys_conv, user_conv = build_grounded_prompt("where is yarn used", sample_evidence.chunks, style="conversational")
    assert "fluent" in sys_conv.lower() or "direct" in sys_conv.lower()
    assert "where is yarn used" in user_conv
    assert "[E1]" in user_conv

    # Executive
    sys_exec, user_exec = build_grounded_prompt("where is yarn used", sample_evidence.chunks, style="executive")
    assert "bullet" in sys_exec.lower() or "bullet" in user_exec.lower()

    # Concise
    sys_conc, user_conc = build_grounded_prompt("where is yarn used", sample_evidence.chunks, style="concise")
    assert "concise" in sys_conc.lower() or "concise" in user_conc.lower()


# ---------------------------------------------------------------------------
# Test 6: Timeout simulation triggering deterministic fallback
# ---------------------------------------------------------------------------
def test_offline_timeout_triggers_deterministic_fallback(sample_evidence):
    provider = MockLLMProvider(should_timeout=True)
    config = LLMConfig(provider="mock", fallback_to_deterministic=True)
    synthesizer = GenerativeRAGSynthesizer(llm_config=config, llm_provider=provider)

    result = synthesizer.synthesize("where is yarn used", sample_evidence)

    assert result.fallback_occurred is True
    assert "timed out" in (result.fallback_reason or "").lower() or "timeout" in (result.fallback_reason or "").lower()
    assert result.provider_used == "deterministic_grounded_engine"
    assert "YARN" in result.answer


# ---------------------------------------------------------------------------
# Test 7: Connection error simulation triggering deterministic fallback
# ---------------------------------------------------------------------------
def test_offline_connection_failure_triggers_deterministic_fallback(sample_evidence):
    provider = MockLLMProvider(should_fail_connection=True)
    config = LLMConfig(provider="mock", fallback_to_deterministic=True)
    synthesizer = GenerativeRAGSynthesizer(llm_config=config, llm_provider=provider)

    result = synthesizer.synthesize("where is yarn used", sample_evidence)

    assert result.fallback_occurred is True
    assert "connection" in (result.fallback_reason or "").lower() or "mock connection refused" in (result.fallback_reason or "").lower()
    assert result.provider_used == "deterministic_grounded_engine"
    assert "YARN" in result.answer


# ---------------------------------------------------------------------------
# Test 8: Truthful reporting
# ---------------------------------------------------------------------------
def test_truthful_provider_accounting(sample_evidence):
    # Fallback case: provider_used must reflect deterministic engine, NEVER mock or ollama
    fail_provider = MockLLMProvider(should_fail_connection=True)
    config_fail = LLMConfig(provider="ollama", fallback_to_deterministic=True)
    synthesizer_fail = GenerativeRAGSynthesizer(llm_config=config_fail, llm_provider=fail_provider)

    result_fail = synthesizer_fail.synthesize("where is yarn used", sample_evidence)
    assert result_fail.provider_used == "deterministic_grounded_engine"
    assert result_fail.fallback_occurred is True

    # Real mock success case: provider_used must be "mock" and fallback_occurred False
    good_provider = MockLLMProvider()
    config_mock = LLMConfig(provider="mock", model="mock-model", fallback_to_deterministic=True)
    synthesizer_good = GenerativeRAGSynthesizer(llm_config=config_mock, llm_provider=good_provider)
    result_good = synthesizer_good.synthesize("where is yarn used", sample_evidence)
    assert result_good.provider_used == "mock"
    assert result_good.fallback_occurred is False


# ---------------------------------------------------------------------------
# Test 9: Monotonic stage latency accounting
# ---------------------------------------------------------------------------
def test_latency_monotonic_accounting(sample_evidence):
    provider = MockLLMProvider()
    config = LLMConfig(provider="mock", fallback_to_deterministic=True)
    synthesizer = GenerativeRAGSynthesizer(llm_config=config, llm_provider=provider)

    result = synthesizer.synthesize("where is yarn used", sample_evidence)

    assert result.generation_time_ms >= 0.0
    stage_latencies = result.metadata.get("stage_latencies_ms", {})
    assert "synthesis_ms" in stage_latencies
    assert "validation_ms" in stage_latencies
    assert "dedup_ms" in stage_latencies
    assert stage_latencies["synthesis_ms"] >= 0.0
    assert stage_latencies["validation_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Test 10: Parsing structured LLM output formats
# ---------------------------------------------------------------------------
def test_parse_structured_llm_output_various_formats():
    # Markdown fenced json with string claims
    fenced_raw = "```json\n{\"answer\": \"YARN schedules workloads [E1].\", \"claims\": [\"YARN schedules workloads [E1].\"]}\n```"
    ans, claims, abstained, reason = parse_structured_llm_output(fenced_raw)
    assert ans == "YARN schedules workloads [E1]."
    assert len(claims) == 1
    assert abstained is False

    # Raw json with structured dict claim
    raw_json = "{\"answer\": \"Direct answer [E1].\", \"claims\": [{\"text\": \"Direct answer\", \"evidence_ids\": [\"E1\"]}]}"
    ans, claims, abstained, reason = parse_structured_llm_output(raw_json)
    assert ans == "Direct answer [E1]."
    assert len(claims) == 1
    assert abstained is False

    # Plain text fallback
    plain_text = "YARN is used to manage cluster resources [E1]. It coordinates batch and stream engines [E2]."
    ans, claims, abstained, reason = parse_structured_llm_output(plain_text)
    assert ans == plain_text
    assert len(claims) >= 1
    assert abstained is False

    # Insufficient evidence
    ans, claims, abstained, reason = parse_structured_llm_output("INSUFFICIENT_EVIDENCE")
    assert abstained is True


# ---------------------------------------------------------------------------
# Test 11: Fast offline Ollama availability check
# ---------------------------------------------------------------------------
def test_ollama_provider_fast_offline_availability_check():
    # Points to a closed local port
    config = LLMConfig(provider="ollama", base_url="http://127.0.0.1:59999/v1", timeout_seconds=1.0)
    provider = OllamaProvider(config)

    t0 = time.monotonic()
    is_avail, reason = provider.is_available()
    elapsed = time.monotonic() - t0

    assert is_avail is False
    assert reason is not None
    assert elapsed < 2.5


# ---------------------------------------------------------------------------
# Test 12: FastAPI status endpoint metadata
# ---------------------------------------------------------------------------
def test_api_status_llm_metadata():
    startup_event()
    client = TestClient(app)

    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()

    assert "llm" in data
    llm_info = data["llm"]
    assert "configured_provider" in llm_info
    assert "effective_provider" in llm_info
    assert "fallback_enabled" in llm_info
    assert "model" in llm_info


# ---------------------------------------------------------------------------
# Test 13: FastAPI search with mock provider override
# ---------------------------------------------------------------------------
def test_api_search_with_mock_provider_override():
    import api
    api.startup_event()
    api.pipeline.add_document(
        "Yet Another Resource Negotiator (YARN) functions as the cluster operating system, "
        "separating resource management from job scheduling across execution frameworks. "
        "Across distributed clusters, YARN schedules diverse heterogeneous workloads including "
        "MapReduce batch processing, Spark SQL analytics, and Flink streaming reliably.",
        source="test_yarn_doc.txt"
    )
    client = TestClient(api.app)

    req = {
        "query": "where is yarn used",
        "llm_provider": "mock",
        "top_k": 2
    }
    response = client.post("/api/search", json=req)
    assert response.status_code == 200
    data = response.json()

    assert "grounded_answer" in data
    ga = data["grounded_answer"]
    assert ga["provider_used"] == "mock"
    assert ga["fallback_occurred"] is False
    assert len(data["evidence"]) > 0


# ---------------------------------------------------------------------------
# Test 14: FastAPI search with deterministic override
# ---------------------------------------------------------------------------
def test_api_search_with_deterministic_override():
    import api
    api.startup_event()
    api.pipeline.add_document(
        "Yet Another Resource Negotiator (YARN) functions as the cluster operating system, "
        "separating resource management from job scheduling across execution frameworks. "
        "Across distributed clusters, YARN schedules diverse heterogeneous workloads including "
        "MapReduce batch processing, Spark SQL analytics, and Flink streaming reliably.",
        source="test_yarn_doc.txt"
    )
    client = TestClient(api.app)

    req = {
        "query": "where is yarn used",
        "llm_provider": "deterministic",
        "top_k": 2
    }
    response = client.post("/api/search", json=req)
    assert response.status_code == 200
    data = response.json()

    assert "grounded_answer" in data
    ga = data["grounded_answer"]
    assert ga["provider_used"] == "deterministic_grounded_engine"
