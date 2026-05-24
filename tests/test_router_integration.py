"""Integration tests for the full answer_question pipeline.

These tests mock answer_department at the boundary so we exercise
routing, splitting, vague detection, and answer assembly logic
together without needing a live LLM or Neo4j instance.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

from langchain_core.documents import Document

from app.agents.router import answer_question

FAKE_HR_DOC = Document(
    page_content="\ntext: Full-time employees accrue 15 days of PTO per year.",
    metadata={"source": "HR-001", "title": "PTO Policy", "department": "hr"},
)
FAKE_FIN_DOC = Document(
    page_content="\ntext: NYC hotel budget must not exceed $350 per night.",
    metadata={"source": "FIN-002", "title": "Corporate Travel Policy", "department": "finance"},
)


def _fake_hr(*_args: Any, **_kwargs: Any) -> tuple[str, list[Document]]:
    return "Full-time employees accrue 15 days of PTO per year.", [FAKE_HR_DOC]


def _fake_fin(*_args: Any, **_kwargs: Any) -> tuple[str, list[Document]]:
    return "NYC hotels must not exceed $350 per night.", [FAKE_FIN_DOC]


def _fake_answer_department(question: str, department: str, mode: str = "fast") -> tuple[str, list[Document]]:
    if department == "hr":
        return _fake_hr()
    return _fake_fin()


# ── Single-department routing ────────────────────────────────────────


@patch("app.agents.router.answer_department", side_effect=_fake_answer_department)
def test_explicit_hr_department_routes_directly(mock_ad: Any) -> None:
    result = answer_question("How much PTO do I accrue?", department="hr")

    assert result["department_routed"] == "hr"
    assert "15 days" in result["answer"]
    mock_ad.assert_called_once()


@patch("app.agents.router.answer_department", side_effect=_fake_answer_department)
def test_explicit_finance_department_routes_directly(mock_ad: Any) -> None:
    result = answer_question("What is the hotel budget?", department="finance")

    assert result["department_routed"] == "finance"
    assert "$350" in result["answer"]
    mock_ad.assert_called_once()


# ── Vague question detection ────────────────────────────────────────


def test_vague_question_returns_clarification() -> None:
    """Vague questions without department context should ask for clarification."""
    result = answer_question("How much do I get?")

    assert "clarify" in result["answer"].lower()
    assert result["sources"] == []
    assert result["department_routed"] == "both"


def test_vague_question_with_explicit_department_does_not_clarify() -> None:
    """When department is explicit, vague questions skip the clarification guard."""
    with patch("app.agents.router.answer_department", side_effect=_fake_answer_department):
        result = answer_question("How much do I get?", department="hr")

    assert "clarify" not in result["answer"].lower()
    assert result["department_routed"] == "hr"


# ── Multi-question splitting ────────────────────────────────────────


@patch("app.agents.router.answer_department", side_effect=_fake_answer_department)
def test_multi_question_splits_and_answers_each(mock_ad: Any) -> None:
    result = answer_question(
        "How much PTO do I accrue? What is the hotel budget for NYC?"
    )

    assert result["department_routed"] == "both"
    assert "1." in result["answer"]
    assert "2." in result["answer"]
    assert len(result["sources"]) >= 2


@patch("app.agents.router.answer_department", side_effect=_fake_answer_department)
def test_multi_question_sources_are_deduped(mock_ad: Any) -> None:
    result = answer_question(
        "How much PTO do I accrue? How much vacation do I accrue?"
    )

    assert [source["source"] for source in result["sources"]] == ["HR-001"]


@patch("app.agents.router.answer_department", side_effect=_fake_answer_department)
def test_multi_question_skips_vague_subquestions(mock_ad: Any) -> None:
    """Vague subquestions in a bundle should get clarification, not retrieval."""
    result = answer_question(
        "How much PTO do I accrue? How much do I get?"
    )

    assert "1." in result["answer"]
    assert "2." in result["answer"]
    assert "clarify" in result["answer"].lower()


# ── Mode passthrough ────────────────────────────────────────────────


@patch("app.agents.router.answer_department", side_effect=_fake_answer_department)
def test_mode_is_passed_through(mock_ad: Any) -> None:
    answer_question("How much PTO do I accrue?", department="hr", mode="deep")

    _, kwargs = mock_ad.call_args
    # mode should be passed via positional or keyword
    call_args = mock_ad.call_args
    assert "deep" in str(call_args)
