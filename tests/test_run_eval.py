from __future__ import annotations

from scripts.run_eval import missing_required_terms, present_forbidden_terms, source_ids


# ── missing_required_terms ───────────────────────────────────────────


def test_missing_required_terms_is_case_insensitive() -> None:
    answer = "Employees receive 15 days of PTO and later 20 days."

    assert missing_required_terms(answer, ["15 DAYS", "20 days"]) == []
    assert missing_required_terms(answer, ["$350"]) == ["$350"]


def test_missing_required_terms_returns_all_missing() -> None:
    answer = "The policy covers PTO."
    missing = missing_required_terms(answer, ["15 days", "$350", "8 weeks"])
    assert missing == ["15 days", "$350", "8 weeks"]


def test_missing_required_terms_empty_list() -> None:
    assert missing_required_terms("any answer", []) == []


# ── present_forbidden_terms ──────────────────────────────────────────


def test_present_forbidden_terms_is_case_insensitive() -> None:
    answer = "Please clarify what policy topic this question refers to."

    assert present_forbidden_terms(answer, ["CLARIFY"]) == ["CLARIFY"]
    assert present_forbidden_terms(answer, ["15 days"]) == []


def test_present_forbidden_terms_catches_multiple() -> None:
    answer = "You get 30 days per year and $350 for hotels."
    found = present_forbidden_terms(answer, ["30 days", "$350", "8 weeks"])
    assert found == ["30 days", "$350"]


def test_present_forbidden_terms_empty_list() -> None:
    assert present_forbidden_terms("any answer", []) == []


# ── source_ids ───────────────────────────────────────────────────────


def test_source_ids_extracts_ids() -> None:
    response = {
        "sources": [
            {"source": "HR-001", "title": "PTO"},
            {"source": "FIN-002", "title": "Travel"},
        ]
    }
    assert source_ids(response) == {"HR-001", "FIN-002"}


def test_source_ids_handles_missing_source_key() -> None:
    """Sources without a 'source' key should be skipped."""
    assert source_ids({"sources": [{"title": "PTO"}]}) == set()


def test_source_ids_handles_none_source_value() -> None:
    """Sources with source=None should be skipped."""
    assert source_ids({"sources": [{"source": None, "title": "PTO"}]}) == set()


def test_source_ids_handles_empty_sources_list() -> None:
    assert source_ids({"sources": []}) == set()


def test_source_ids_handles_missing_sources_key() -> None:
    assert source_ids({}) == set()

