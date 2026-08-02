"""Semantic answer cache, partitioned by the caller's access.

A semantic cache keyed on the question alone is a data-leak bug in any system
with per-user authorization. "What are the M&A approval thresholds?" asked by an
executive produces an answer built from `executives`-only documents; cached
under the question text, the next employee to ask a *similar* question is served
that answer directly, with retrieval — and therefore the ACL filter — skipped
entirely. Nothing in the request path would notice.

So the key is `(embedding, exact group set)`, and entries are stored in
per-group-set partitions rather than filtered after lookup. Filtering after
lookup would still require the leaky entry to be found, compared, and rejected
correctly every time; partitioning makes the wrong answer unreachable.

Keying on the *exact* group set rather than something cleverer is deliberate.
A caller holding `{all-employees, finance-team}` cannot reuse an entry created
for `{all-employees}` even though it would be safe to, which costs hit rate.
The alternative — reasoning about which subsets are safe to share — is a
correctness argument that has to hold for every future group, and getting it
wrong is silent.

**Ceilings.** In-process, so each worker keeps its own copy and hit rate falls
as workers scale; a shared Redis is the upgrade. Entries expire on a TTL rather
than on corpus change, so re-ingesting a revised policy leaves stale answers
served for up to the TTL — `invalidate_all()` is what `scripts/ingest.py` should
call once this is deployed anywhere real.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Cosine similarity above which two questions are treated as the same question.
#: High on purpose: "What is the hotel cap in NYC?" and "What is the hotel cap
#: internationally?" are close in embedding space and have different answers, so
#: a permissive threshold serves confidently wrong numbers.
DEFAULT_THRESHOLD = 0.97
DEFAULT_TTL_SECONDS = 3600.0
DEFAULT_MAX_ENTRIES_PER_PARTITION = 256


@dataclass
class CacheEntry:
    embedding: np.ndarray
    question: str
    payload: dict[str, Any]
    created: float


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    partitions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def partition_key(groups: frozenset[str] | None) -> str:
    """Canonical name for an access partition.

    `None` means ACL enforcement is off (offline tooling) and gets its own
    partition so its entries can never be served to an authenticated caller.
    """
    if groups is None:
        return "\x00unfiltered"
    return "\x00".join(sorted(groups))


class SemanticCache:
    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES_PER_PARTITION,
    ) -> None:
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._partitions: dict[str, list[CacheEntry]] = {}
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def _fresh(self, entries: list[CacheEntry], now: float) -> list[CacheEntry]:
        return [e for e in entries if now - e.created < self.ttl_seconds]

    def get(self, embedding: np.ndarray, groups: frozenset[str] | None) -> dict[str, Any] | None:
        key = partition_key(groups)
        now = time.monotonic()
        with self._lock:
            entries = self._fresh(self._partitions.get(key, []), now)
            self._partitions[key] = entries
            if not entries:
                self.stats.misses += 1
                return None
            matrix = np.array([e.embedding for e in entries])
            scores = matrix @ embedding
            best = int(np.argmax(scores))
            if scores[best] >= self.threshold:
                self.stats.hits += 1
                return dict(entries[best].payload)
            self.stats.misses += 1
            return None

    def put(
        self,
        embedding: np.ndarray,
        groups: frozenset[str] | None,
        question: str,
        payload: dict[str, Any],
    ) -> None:
        key = partition_key(groups)
        now = time.monotonic()
        with self._lock:
            entries = self._fresh(self._partitions.get(key, []), now)
            entries.append(
                CacheEntry(embedding=embedding, question=question, payload=dict(payload), created=now)
            )
            if len(entries) > self.max_entries:
                # Oldest-first. A cache this small does not justify tracking
                # recency; if hit rate ever matters, Redis with LRU replaces this.
                del entries[0 : len(entries) - self.max_entries]
                self.stats.evictions += 1
            self._partitions[key] = entries
            self.stats.partitions = len(self._partitions)

    def invalidate_all(self) -> None:
        """Drop everything. Call after the corpus changes."""
        with self._lock:
            self._partitions.clear()
            self.stats.partitions = 0

    def size(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._partitions.values())


def normalize(vector: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    return array / norm if norm else array


#: Process-wide instance. See the ceilings note in the module docstring.
_cache: SemanticCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                from app.config import settings

                _cache = SemanticCache(
                    threshold=settings.semantic_cache_threshold,
                    ttl_seconds=settings.semantic_cache_ttl_seconds,
                )
    return _cache
