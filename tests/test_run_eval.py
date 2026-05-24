from __future__ import annotations

from scripts.run_eval import missing_required_terms, present_forbidden_terms


def test_missing_required_terms_is_case_insensitive() -> None:
    answer = "Employees receive 15 days of PTO and later 20 days."

    assert missing_required_terms(answer, ["15 DAYS", "20 days"]) == []
    assert missing_required_terms(answer, ["$350"]) == ["$350"]


def test_present_forbidden_terms_is_case_insensitive() -> None:
    answer = "Please clarify what policy topic this question refers to."

    assert present_forbidden_terms(answer, ["CLARIFY"]) == ["CLARIFY"]
    assert present_forbidden_terms(answer, ["15 days"]) == []
