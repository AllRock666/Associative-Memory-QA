"""
Test suite for Pasted Chat Paragraphs, Random Out-of-Domain Questions,
Targeted In-Domain Question Answering, and Dynamic Mind Map Generation.
"""

import pytest
from fastapi.testclient import TestClient
import api


def test_pasted_chat_workflow():
    api.startup_event()
    client = TestClient(api.app)

    chat_id = "test_chat_pasted_123"

    # Step 1: User pastes a large paragraph into chat
    large_paragraph = (
        "Quantum computing harnesses the unique properties of quantum mechanics, such as superposition, "
        "entanglement, and interference, to process information exponentially faster than classical computers "
        "for specific problem classes. In a classical computer, bits represent either a 0 or a 1. In contrast, "
        "quantum bits (qubits) can exist in superpositions of states |0> and |1>. Shor's algorithm provides "
        "polynomial-time integer factorization, posing a significant challenge to RSA public-key cryptography, "
        "while Grover's algorithm provides quadratic speedup for unstructured database searching. Quantum "
        "decoherence and environmental noise remain the primary physical obstacles to scaling fault-tolerant "
        "quantum computing hardware."
    )

    # 1. Send large paragraph into chat
    res1 = client.post(
        "/api/search",
        json={"query": large_paragraph, "chat_id": chat_id}
    )
    assert res1.status_code == 200

    # 2. Ask random unrelated question (out-of-domain) with web search enabled
    res_random = client.post(
        "/api/search",
        json={"query": "What is the capital of France?", "chat_id": chat_id, "web_search_enabled": True}
    )
    assert res_random.status_code == 200
    r_data = res_random.json()
    assert r_data.get("web_search_used") is True or r_data.get("cached_knowledge_used") is True
    assert "paris" in r_data.get("answer", "").lower()
    assert len(r_data.get("web_sources", [])) > 0

    # Also verify that when web search is explicitly disabled, it abstains without hallucinating
    res_offline_random = client.post(
        "/api/search",
        json={"query": "What is the capital of France?", "chat_id": chat_id, "web_search_enabled": False}
    )
    assert res_offline_random.status_code == 200
    off_answer = res_offline_random.json()["answer"].lower()
    assert any(term in off_answer for term in ["cannot answer", "insufficient evidence", "not found", "no relevant", "do not contain"])

    # 3. Ask specific questions on the paragraph sent
    res_q1 = client.post(
        "/api/search",
        json={"query": "What speedup does Grover's algorithm provide?", "chat_id": chat_id}
    )
    assert res_q1.status_code == 200
    ans_q1 = res_q1.json()["answer"]
    assert "quadratic" in ans_q1.lower()
    assert res_q1.json().get("web_search_used") is False

    res_q2 = client.post(
        "/api/search",
        json={"query": "What are the primary physical obstacles mentioned?", "chat_id": chat_id}
    )
    assert res_q2.status_code == 200
    ans_q2 = res_q2.json()["answer"].lower()
    assert "decoherence" in ans_q2 or "noise" in ans_q2

    # 4. Ask to make a mind map of that paragraph
    res_mm = client.post(
        "/api/search",
        json={"query": "Make a mind map of that paragraph", "chat_id": chat_id}
    )
    assert res_mm.status_code == 200
    mm_data = res_mm.json()
    assert "visualization" in mm_data or "mindmap" in mm_data["answer"]
    content = mm_data.get("visualization", {}).get("content", "") or mm_data["answer"]
    content_lower = content.lower()
    assert "mindmap" in content_lower
    # Ensure mindmap contains actual concepts from the provided paragraph
    assert any(term in content_lower for term in ["quantum", "superposition", "qubits", "shor", "grover", "decoherence"])
    # Ensure NO fake/hardcoded dummy nodes
    assert "concept alpha" not in content_lower
