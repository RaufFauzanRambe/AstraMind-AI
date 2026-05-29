"""
AstraMind AI - Memory Manager
===============================
Manages both short-term (conversation) and long-term (persistent) memory
for the AstraMind system. Supports memory retrieval, storage, and decay.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from .config import MemoryConfig

logger = logging.getLogger(__name__)


class MemoryEntry:
    """A single memory entry with metadata."""

    def __init__(
        self,
        content: str,
        entry_type: str = "interaction",
        timestamp: Optional[float] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.entry_type = entry_type
        self.timestamp = timestamp or time.time()
        self.importance = importance
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_accessed = self.timestamp

    def access(self) -> None:
        """Mark this memory as accessed, increasing its relevance."""
        self.access_count += 1
        self.last_accessed = time.time()

    @property
    def effective_importance(self) -> float:
        """Calculate effective importance considering access frequency and recency."""
        recency_factor = 1.0 / (1.0 + (time.time() - self.timestamp) / 86400)  # Decay over days
        access_factor = min(1.0 + self.access_count * 0.1, 2.0)
        return self.importance * recency_factor * access_factor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "entry_type": self.entry_type,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        entry = cls(
            content=data["content"],
            entry_type=data.get("entry_type", "interaction"),
            timestamp=data.get("timestamp"),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {}),
        )
        entry.access_count = data.get("access_count", 0)
        entry.last_accessed = data.get("last_accessed", entry.timestamp)
        return entry


class ShortTermMemory:
    """
    Short-term memory for the current conversation context.
    Stores recent interactions with a configurable maximum size.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._entries: List[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        """Add an entry to short-term memory, evicting oldest if at capacity."""
        self._entries.append(entry)
        if len(self._entries) > self.max_size:
            evicted = self._entries.pop(0)
            logger.debug(f"Evicted from short-term memory: {evicted.content[:50]}...")

    def get_recent(self, count: int = 10) -> List[MemoryEntry]:
        """Get the most recent entries."""
        return self._entries[-count:]

    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Simple keyword-based search in short-term memory."""
        query_terms = set(query.lower().split())
        scored = []
        for entry in self._entries:
            content_terms = set(entry.content.lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                scored.append((entry, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:top_k]]

    def clear(self) -> None:
        """Clear all short-term memory."""
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)


class LongTermMemory:
    """
    Long-term persistent memory that survives across sessions.
    Uses a JSON file store with importance-based retention.
    """

    def __init__(self, store_path: str, max_size: int = 1000, decay_rate: float = 0.95):
        self.store_path = store_path
        self.max_size = max_size
        self.decay_rate = decay_rate
        self._entries: List[MemoryEntry] = []
        self._loaded = False

    async def load(self) -> None:
        """Load memory entries from the persistent store."""
        if self._loaded:
            return

        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._entries = [MemoryEntry.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self._entries)} long-term memory entries.")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load memory store: {e}")
                self._entries = []
        else:
            logger.info("No existing memory store found. Starting fresh.")
            self._entries = []

        self._loaded = True

    async def save(self) -> None:
        """Save memory entries to the persistent store."""
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        try:
            data = [e.to_dict() for e in self._entries]
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self._entries)} long-term memory entries.")
        except IOError as e:
            logger.error(f"Failed to save memory store: {e}")

    def add(self, entry: MemoryEntry) -> None:
        """Add an entry to long-term memory with importance-based eviction."""
        self._entries.append(entry)
        if len(self._entries) > self.max_size:
            self._evict_lowest_importance()

    def _evict_lowest_importance(self) -> None:
        """Evict the entry with the lowest effective importance."""
        if not self._entries:
            return
        min_idx = min(range(len(self._entries)), key=lambda i: self._entries[i].effective_importance)
        evicted = self._entries.pop(min_idx)
        logger.debug(f"Evicted from long-term memory: {evicted.content[:50]}...")

    def apply_decay(self) -> None:
        """Apply decay to all long-term memory entries."""
        for entry in self._entries:
            entry.importance *= self.decay_rate

    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Search long-term memory by keyword relevance and importance."""
        query_terms = set(query.lower().split())
        scored = []
        for entry in self._entries:
            content_terms = set(entry.content.lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                score = overlap * entry.effective_importance
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for entry, _ in scored[:top_k]:
            entry.access()
            results.append(entry)

        return results

    @property
    def size(self) -> int:
        return len(self._entries)


class MemoryManager:
    """
    Unified memory manager that coordinates short-term and long-term memory.

    Provides a single interface for storing and retrieving memories,
    managing the transfer of important short-term memories to long-term storage,
    and maintaining conversation context.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.short_term = ShortTermMemory(max_size=config.max_short_term_memory)
        self.long_term = LongTermMemory(
            store_path=config.memory_store_path,
            max_size=config.max_long_term_memory,
            decay_rate=config.memory_decay_rate,
        )
        self._conversation_memories: Dict[str, ShortTermMemory] = {}

    async def initialize(self) -> None:
        """Initialize the memory system, loading persistent storage."""
        await self.long_term.load()
        logger.info("Memory manager initialized.")

    async def store_interaction(
        self,
        user_input: str,
        response: str,
        conversation_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a user-AI interaction in both short-term and long-term memory.

        Args:
            user_input: The user's input text.
            response: The AI's response text.
            conversation_id: Optional conversation identifier.
            importance: The importance score (0.0-1.0).
            metadata: Additional metadata for the interaction.
        """
        combined = f"User: {user_input}\nAstraMind: {response}"
        entry = MemoryEntry(
            content=combined,
            entry_type="interaction",
            importance=importance,
            metadata={
                "conversation_id": conversation_id,
                **(metadata or {}),
            },
        )

        # Store in short-term memory
        self.short_term.add(entry)

        # Store in conversation-specific memory
        if conversation_id:
            if conversation_id not in self._conversation_memories:
                self._conversation_memories[conversation_id] = ShortTermMemory(
                    max_size=self.config.max_short_term_memory
                )
            self._conversation_memories[conversation_id].add(entry)

        # Store important interactions in long-term memory
        if importance >= 0.6:
            self.long_term.add(entry)

    async def retrieve_relevant(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for a given query.

        Searches both short-term and long-term memory, merging and
        ranking results by relevance and importance.

        Args:
            query: The query to search for.
            conversation_id: Optional conversation to prioritize.
            top_k: Maximum number of results to return.

        Returns:
            List of memory entry dictionaries.
        """
        results = []

        # Search conversation-specific memory first
        if conversation_id and conversation_id in self._conversation_memories:
            conv_results = self._conversation_memories[conversation_id].search(query, top_k=top_k)
            for entry in conv_results:
                results.append({**entry.to_dict(), "source": "conversation"})

        # Search short-term memory
        stm_results = self.short_term.search(query, top_k=top_k)
        for entry in stm_results:
            if entry.to_dict() not in [r for r in results if r.get("source") == "conversation"]:
                results.append({**entry.to_dict(), "source": "short_term"})

        # Search long-term memory
        ltm_results = self.long_term.search(query, top_k=top_k)
        for entry in ltm_results:
            results.append({**entry.to_dict(), "source": "long_term"})

        # Sort by effective importance and return top_k
        results.sort(key=lambda x: x.get("importance", 0), reverse=True)
        return results[:top_k]

    async def save(self) -> None:
        """Save all persistent memory."""
        await self.long_term.save()

    def stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "short_term_size": self.short_term.size,
            "long_term_size": self.long_term.size,
            "active_conversations": len(self._conversation_memories),
        }
