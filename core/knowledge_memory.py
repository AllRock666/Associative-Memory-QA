"""
Persistent Source-Backed Knowledge Memory Module
================================================
Implements persistent Layer A (Source-backed knowledge memory) with:
1. Stable memory IDs and provenance tracing (URL, doc ID, titles).
2. Content-hash deduplication (SHA-256).
3. Creation and last-verification timestamps.
4. Configurable TTL (expiry) and active invalidation.
5. JSON persistence to data/web_memory_store.json.
6. Semantic retrieval integration for instant cache hits without re-querying the web.
"""

import os
import re
import json
import time
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.schemas import EvidenceChunk

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeMemoryEntry:
    """A validated unit of factual knowledge persisted in memory."""
    memory_id: str
    text: str
    source_type: str = "web"  # "web" | "uploaded_document" | "cached_knowledge"
    source: str = ""
    url: Optional[str] = None
    title: Optional[str] = None
    published_date: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_verified_at: float = field(default_factory=time.time)
    content_hash: str = ""
    version: int = 1
    confidence: float = 1.0
    is_validated: bool = True
    ttl_seconds: Optional[float] = 86400 * 7  # 7 days default retention
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Determines if the entry has passed its TTL."""
        if self.ttl_seconds is None:
            return False
        curr = now or time.time()
        return (curr - self.last_verified_at) > self.ttl_seconds

    def to_evidence_chunk(self, rank: int = 1, score: float = 0.90) -> EvidenceChunk:
        """Converts knowledge entry into a standard EvidenceChunk for retrieval & synthesis."""
        return EvidenceChunk(
            chunk_id=self.memory_id,
            doc_id=self.url or self.source,
            source=self.title or self.source,
            text=self.text,
            similarity_score=score,
            hopfield_attention=score,
            rank=rank,
            is_core_attractor=True,
            source_type="cached_knowledge" if self.source_type == "web" else self.source_type,
            url=self.url,
            title=self.title,
            published_date=self.published_date,
            content_hash=self.content_hash,
            metadata={
                **self.metadata,
                "cached_memory_id": self.memory_id,
                "created_at": self.created_at,
                "last_verified_at": self.last_verified_at
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeMemoryEntry":
        return cls(**data)


class PersistentKnowledgeMemory:
    """
    Persistent storage and indexing engine for source-backed knowledge chunks.
    Separates persistent factual entries from transient multi-turn conversation memory.
    """

    def __init__(self, persistence_path: Optional[str] = None, filepath: Optional[str] = None, enabled: bool = True):
        self.persistence_path = Path(filepath or persistence_path or "data/web_memory_store.json")
        self.enabled = enabled
        self._entries: Dict[str, KnowledgeMemoryEntry] = {}
        self._hash_to_id: Dict[str, str] = {}
        self._url_to_ids: Dict[str, List[str]] = {}
        self.load()

    def add_entry(
        self,
        text: str,
        source: str,
        url: Optional[str] = None,
        title: Optional[str] = None,
        published_date: Optional[str] = None,
        source_type: str = "web",
        ttl_seconds: Optional[float] = 86400 * 7,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeMemoryEntry:
        """
        Stores or updates a knowledge chunk with content-hash deduplication.
        If identical content already exists, updates last_verified_at.
        """
        clean_text = text.strip()
        chash = hashlib.sha256(clean_text.encode("utf-8", errors="ignore")).hexdigest()

        # Deduplication check by content hash
        if chash in self._hash_to_id:
            existing_id = self._hash_to_id[chash]
            existing = self._entries[existing_id]
            existing.last_verified_at = time.time()
            if url and not existing.url:
                existing.url = url
            if title and not existing.title:
                existing.title = title
            self.save()
            return existing

        # Generate stable, clean memory ID
        mem_id = f"mem_{source_type[:3]}_{chash[:12]}"
        now = time.time()

        entry = KnowledgeMemoryEntry(
            memory_id=mem_id,
            text=clean_text,
            source_type=source_type,
            source=source,
            url=url,
            title=title,
            published_date=published_date,
            created_at=now,
            last_verified_at=now,
            content_hash=chash,
            confidence=confidence,
            ttl_seconds=ttl_seconds,
            metadata=metadata or {}
        )

        self._entries[mem_id] = entry
        self._hash_to_id[chash] = mem_id
        if url:
            self._url_to_ids.setdefault(url, []).append(mem_id)

        self.save()
        return entry

    def get_valid_entries(self) -> List[KnowledgeMemoryEntry]:
        """Returns all non-expired entries in the knowledge store."""
        now = time.time()
        return [e for e in self._entries.values() if not e.is_expired(now)]

    def search_knowledge(self, query: str, top_k: int = 5) -> List[EvidenceChunk]:
        """
        Fast lexical & keyword retrieval over cached knowledge.
        Provides zero-latency cache hits when relevant entries already exist.
        """
        valid_entries = self.get_valid_entries()
        if not valid_entries:
            return []

        q_words = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower()))
        scored = []
        for entry in valid_entries:
            e_text_lower = entry.text.lower()
            e_title_lower = (entry.title or "").lower()
            # Token match count
            overlap = sum(1 for w in q_words if w in e_text_lower or w in e_title_lower)
            if overlap > 0:
                score = overlap / max(len(q_words), 1)
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, entry) in enumerate(scored[:top_k], 1):
            results.append(entry.to_evidence_chunk(rank=rank, score=round(0.80 + score * 0.18, 4)))
        return results

    def invalidate_url(self, url: str) -> int:
        """Removes all stored knowledge chunks originating from a given URL."""
        ids_to_remove = self._url_to_ids.pop(url, [])
        removed_count = 0
        for mid in ids_to_remove:
            if mid in self._entries:
                entry = self._entries.pop(mid)
                self._hash_to_id.pop(entry.content_hash, None)
                removed_count += 1
        if removed_count > 0:
            self.save()
        return removed_count

    def get_all_entries(self) -> List[KnowledgeMemoryEntry]:
        """Returns all entries in the knowledge store."""
        return list(self._entries.values())

    def delete_entry(self, memory_id: str) -> bool:
        """Deletes a specific knowledge entry by ID."""
        if memory_id not in self._entries:
            return False
        entry = self._entries.pop(memory_id)
        self._hash_to_id.pop(entry.content_hash, None)
        if entry.url and entry.url in self._url_to_ids:
            self._url_to_ids[entry.url] = [i for i in self._url_to_ids[entry.url] if i != memory_id]
            if not self._url_to_ids[entry.url]:
                del self._url_to_ids[entry.url]
        self.save()
        return True

    def clear(self):
        """Clears all stored knowledge entries."""
        self._entries.clear()
        self._hash_to_id.clear()
        self._url_to_ids.clear()
        self.save()

    def save(self):
        """Persists knowledge entries to disk."""
        if not self.enabled:
            return
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "updated_at": time.time(),
                "entries": [e.to_dict() for e in self._entries.values()]
            }
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not persist knowledge memory to {self.persistence_path}: {e}")

    def load(self):
        """Loads persisted knowledge entries from disk."""
        if not self.persistence_path.exists():
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                entries_data = data.get("entries", [])
                for ed in entries_data:
                    entry = KnowledgeMemoryEntry.from_dict(ed)
                    self._entries[entry.memory_id] = entry
                    self._hash_to_id[entry.content_hash] = entry.memory_id
                    if entry.url:
                        self._url_to_ids.setdefault(entry.url, []).append(entry.memory_id)
        except Exception as e:
            logger.warning(f"Could not load knowledge memory from {self.persistence_path}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        valid = self.get_valid_entries()
        return {
            "total_entries": len(self._entries),
            "valid_entries": len(valid),
            "expired_entries": len(self._entries) - len(valid),
            "unique_urls": len(self._url_to_ids),
            "persistence_file": str(self.persistence_path)
        }
