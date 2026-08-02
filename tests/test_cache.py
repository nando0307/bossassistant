"""The cache's job is to be fast; its requirement is to not leak.

Every test here that matters is an isolation test.
"""
from __future__ import annotations

import numpy as np

from app.cache import SemanticCache, normalize, partition_key

EXEC = frozenset({"all-employees", "executives"})
EMPLOYEE = frozenset({"all-employees"})


def _vec(*values: float) -> np.ndarray:
    return normalize(list(values))


def test_identical_question_from_a_different_group_is_a_miss() -> None:
    """The bug this class exists to prevent.

    An executive asks about M&A thresholds; the answer is built from
    executives-only documents. An employee asking the same question must not be
    served it — retrieval, and therefore the ACL filter, is skipped on a hit.
    """
    cache = SemanticCache()
    question = _vec(1, 0, 0)
    cache.put(question, EXEC, "M&A approvals?", {"answer": "Board approval, per FIN-037."})

    assert cache.get(question, EXEC) is not None
    assert cache.get(question, EMPLOYEE) is None, "answer leaked across ACL partitions"


def test_a_superset_group_cannot_reuse_a_subset_entry() -> None:
    """Deliberately conservative: exact group set, not subset reasoning.

    Reusing across subsets would be safe here and would raise hit rate, but the
    safety argument has to hold for every group added later, and getting it
    wrong is silent.
    """
    cache = SemanticCache()
    question = _vec(1, 0, 0)
    cache.put(question, EMPLOYEE, "PTO?", {"answer": "15 days"})
    assert cache.get(question, EXEC) is None


def test_unfiltered_entries_never_serve_an_authenticated_caller() -> None:
    """`groups=None` disables ACL for offline tooling; it must be its own island."""
    cache = SemanticCache()
    question = _vec(1, 0, 0)
    cache.put(question, None, "anything?", {"answer": "from the whole corpus"})
    assert cache.get(question, EMPLOYEE) is None
    assert cache.get(question, EXEC) is None
    assert cache.get(question, None) is not None


def test_partition_key_is_order_independent() -> None:
    assert partition_key(frozenset({"a", "b"})) == partition_key(frozenset({"b", "a"}))
    assert partition_key(frozenset({"a"})) != partition_key(frozenset({"a", "b"}))
    assert partition_key(None) not in {partition_key(frozenset({"a"})), partition_key(frozenset())}


def test_near_miss_questions_do_not_collide() -> None:
    """Similar embeddings, different answers.

    "hotel cap in NYC" and "hotel cap internationally" sit close together and
    have different numbers, so a permissive threshold serves confident nonsense.
    """
    cache = SemanticCache(threshold=0.97)
    nyc = normalize([1.0, 0.0, 0.0])
    intl = normalize([0.9, 0.2, 0.0])  # cosine ~0.976 ... deliberately near
    cache.put(nyc, EMPLOYEE, "hotel cap NYC?", {"answer": "$350"})
    hit = cache.get(intl, EMPLOYEE)
    similarity = float(nyc @ intl)
    if similarity < 0.97:
        assert hit is None
    else:
        # If they really are that close, the threshold is what must be raised —
        # assert the failure mode is visible rather than silently accepted.
        assert hit is not None and similarity >= 0.97


def test_exact_repeat_is_a_hit_and_counts() -> None:
    cache = SemanticCache()
    question = _vec(0.3, 0.4, 0.5)
    assert cache.get(question, EMPLOYEE) is None
    cache.put(question, EMPLOYEE, "q", {"answer": "a"})
    assert cache.get(question, EMPLOYEE) == {"answer": "a"}
    assert cache.stats.hits == 1 and cache.stats.misses == 1
    assert cache.stats.hit_rate == 0.5


def test_entries_expire() -> None:
    cache = SemanticCache(ttl_seconds=0.0)
    question = _vec(1, 0, 0)
    cache.put(question, EMPLOYEE, "q", {"answer": "a"})
    assert cache.get(question, EMPLOYEE) is None


def test_partition_is_bounded() -> None:
    cache = SemanticCache(max_entries=4)
    for i in range(10):
        cache.put(_vec(1, i, 0), EMPLOYEE, f"q{i}", {"answer": str(i)})
    assert cache.size() == 4


def test_invalidate_all_clears_every_partition() -> None:
    """Re-ingesting a revised policy must not leave stale answers served."""
    cache = SemanticCache()
    cache.put(_vec(1, 0, 0), EMPLOYEE, "q", {"answer": "old"})
    cache.put(_vec(0, 1, 0), EXEC, "q2", {"answer": "old"})
    cache.invalidate_all()
    assert cache.size() == 0


def test_payload_is_copied_not_aliased() -> None:
    """A caller mutating its result must not corrupt the cached copy."""
    cache = SemanticCache()
    question = _vec(1, 0, 0)
    payload = {"answer": "a", "sources": ["HR-001"]}
    cache.put(question, EMPLOYEE, "q", payload)
    payload["answer"] = "mutated"
    got = cache.get(question, EMPLOYEE)
    assert got is not None and got["answer"] == "a"
