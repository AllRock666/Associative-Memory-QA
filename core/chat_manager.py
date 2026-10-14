"""
Chat Session Management for HopfieldQA
======================================
Provides multi-chat session handling, conversation history persistence across sessions,
and dual-layer storage (disk persistence in data/chat_sessions.json).
"""

import json
import time
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChatManager:
    """Manages multi-chat conversations and persists them to disk."""

    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = Path(storage_dir)
        self.storage_file = self.storage_dir / "chat_sessions.json"
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load persisted chats from JSON disk storage."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._sessions = data.get("sessions", {})
                    elif isinstance(data, list):
                        self._sessions = {s["id"]: s for s in data if isinstance(s, dict) and "id" in s}
                logger.info(f"Loaded {len(self._sessions)} chat sessions from {self.storage_file}")
            except Exception as e:
                logger.warning(f"Could not load chat sessions from {self.storage_file}: {e}")
                self._sessions = {}

    def _save(self) -> None:
        """Persist all chat sessions to JSON disk storage."""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump({"sessions": self._sessions}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save chat sessions to {self.storage_file}: {e}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List summary of all chat sessions sorted by latest activity."""
        summaries = []
        for s_id, s in self._sessions.items():
            msgs = s.get("messages", [])
            last_content = msgs[-1].get("content", "") if msgs else ""
            preview = (last_content[:80] + "...") if len(last_content) > 80 else last_content
            summaries.append({
                "id": s_id,
                "title": s.get("title", "New Chat"),
                "created_at": s.get("created_at", time.time()),
                "updated_at": s.get("updated_at", time.time()),
                "message_count": len(msgs),
                "preview": preview
            })
        summaries.sort(key=lambda x: x["updated_at"], reverse=True)
        return summaries

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chat session with all messages."""
        return self._sessions.get(session_id)

    def create_session(self, title: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new chat session."""
        s_id = session_id or str(uuid.uuid4())
        now = time.time()
        session = {
            "id": s_id,
            "title": title or "New Chat",
            "created_at": now,
            "updated_at": now,
            "messages": []
        }
        self._sessions[s_id] = session
        self._save()
        return session

    def update_session(self, session_id: str, title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Update session title or attributes."""
        if session_id not in self._sessions:
            return None
        if title:
            self._sessions[session_id]["title"] = title
        self._sessions[session_id]["updated_at"] = time.time()
        self._save()
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save()
            return True
        return False

    def add_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_response: Dict[str, Any],
        auto_title: bool = True
    ) -> Dict[str, Any]:
        """
        Record a complete query-answer turn into the session.
        Auto-generates a clean title from the initial user query if default.
        """
        now = time.time()
        if session_id not in self._sessions:
            self.create_session(session_id=session_id)

        session = self._sessions[session_id]
        msgs = session.setdefault("messages", [])

        # Auto title from first user query
        if auto_title and (session.get("title") in ("New Chat", "", None) or len(msgs) == 0):
            clean_t = user_query.strip().rstrip("?").rstrip(".")
            if len(clean_t) > 42:
                clean_t = clean_t[:39] + "..."
            session["title"] = clean_t.capitalize()

        # Add user message
        user_msg = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": user_query,
            "timestamp": now
        }
        msgs.append(user_msg)

        # Add assistant message
        asst_msg = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": assistant_response.get("answer", ""),
            "timestamp": now,
            "confidence": assistant_response.get("confidence", 0.0),
            "evidence": assistant_response.get("evidence", []),
            "associative_diagnostics": assistant_response.get("associative_diagnostics"),
            "grounded_answer": assistant_response.get("grounded_answer"),
            "latencies": assistant_response.get("latencies"),
            "deduplication": assistant_response.get("deduplication")
        }
        msgs.append(asst_msg)

        session["updated_at"] = now
        self._save()
        return session
