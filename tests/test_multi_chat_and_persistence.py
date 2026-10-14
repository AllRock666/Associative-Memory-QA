"""
Tests for multi-chat sessions, conversation history, and document persistence.
"""
import os
import tempfile
import pytest
from starlette.testclient import TestClient

from core.chat_manager import ChatManager
import api
from api import app, pipeline, chat_manager


def test_chat_manager_crud_and_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cm = ChatManager(storage_dir=tmp_dir)

        # Initially empty
        assert len(cm.list_sessions()) == 0

        # Create session
        sess = cm.create_session(title="Research Chat")
        chat_id = sess["id"]
        assert sess["title"] == "Research Chat"
        assert len(cm.list_sessions()) == 1

        # Add turn
        cm.add_turn(
            session_id=chat_id,
            user_query="What is Hopfield memory?",
            assistant_response={
                "answer": "Modern Hopfield memory is an energy-based associative retrieval system.",
                "confidence": 0.95,
                "evidence": [{"id": "E1", "text": "Hopfield memory retrieval"}]
            }
        )

        sess_updated = cm.get_session(chat_id)
        assert len(sess_updated["messages"]) == 2
        assert sess_updated["messages"][0]["role"] == "user"
        assert sess_updated["messages"][0]["content"] == "What is Hopfield memory?"
        assert sess_updated["messages"][1]["role"] == "assistant"
        assert sess_updated["messages"][1]["content"].startswith("Modern Hopfield")

        # Test persistence across re-initialization
        cm_reloaded = ChatManager(storage_dir=tmp_dir)
        assert len(cm_reloaded.list_sessions()) == 1
        sess_reloaded = cm_reloaded.get_session(chat_id)
        assert sess_reloaded is not None
        assert len(sess_reloaded["messages"]) == 2

        # Test auto-titling when title is default "New Chat"
        sess2 = cm_reloaded.create_session(title="New Chat")
        cm_reloaded.add_turn(
            session_id=sess2["id"],
            user_query="Explain distributed YARN containers in detail please.",
            assistant_response={"answer": "YARN containers represent allocated compute resources."},
        )
        sess2_updated = cm_reloaded.get_session(sess2["id"])
        assert sess2_updated["title"].startswith("Explain distributed yarn")

        # Test delete
        assert cm_reloaded.delete_session(chat_id) is True
        assert cm_reloaded.get_session(chat_id) is None
        assert len(cm_reloaded.list_sessions()) == 1


def test_api_chat_endpoints():
    api.startup_event()
    client = TestClient(app)

    # 1. Create chat
    create_res = client.post("/api/chats", json={"title": "Test Multi-Chat API"})
    assert create_res.status_code == 200
    chat_data = create_res.json()["chat"]
    chat_id = chat_data["id"]
    assert chat_data["title"] == "Test Multi-Chat API"

    # 2. List chats
    list_res = client.get("/api/chats")
    assert list_res.status_code == 200
    chats = list_res.json()["chats"]
    assert any(c["id"] == chat_id for c in chats)

    # 3. Get single chat
    get_res = client.get(f"/api/chats/{chat_id}")
    assert get_res.status_code == 200
    assert get_res.json()["chat"]["id"] == chat_id

    # 4. Search within chat session
    search_res = client.post("/api/search", json={
        "query": "where is yarn used",
        "chat_id": chat_id,
        "k": 2,
        "mode": "hopfield",
        "style": "conversational",
        "llm_provider": "deterministic"
    })
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data.get("chat_id") == chat_id

    # Verify message was persisted into chat session
    get_updated = client.get(f"/api/chats/{chat_id}")
    assert get_updated.status_code == 200
    messages = get_updated.json()["chat"]["messages"]
    assert len(messages) >= 2
    assert messages[0]["role"] == "user"
    assert "where is yarn used" in messages[0]["content"]

    # 5. Delete chat
    del_res = client.delete(f"/api/chats/{chat_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Confirm deleted
    get_deleted = client.get(f"/api/chats/{chat_id}")
    assert get_deleted.status_code == 404


def test_api_documents_endpoint():
    api.startup_event()
    client = TestClient(app)
    docs_res = client.get("/api/documents")
    assert docs_res.status_code == 200
    data = docs_res.json()
    assert "documents" in data
    assert "total_chunks" in data
    assert isinstance(data["documents"], list)
