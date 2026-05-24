from __future__ import annotations

from app.agents.router import (
    is_vague_subquestion,
    route_question_fast,
    split_department_questions_fast,
    split_user_questions,
)


# ── split_user_questions ─────────────────────────────────────────────


def test_split_user_questions_from_question_marks() -> None:
    question = "How much PTO do I accrue? What is the hotel budget for NYC?"

    assert split_user_questions(question) == [
        "How much PTO do I accrue?",
        "What is the hotel budget for NYC?",
    ]


def test_split_user_questions_from_numbered_list() -> None:
    question = "1. How much PTO do I accrue? 2. What is the hotel budget?"

    assert split_user_questions(question) == [
        "How much PTO do I accrue?",
        "What is the hotel budget?",
    ]


def test_split_single_question_returns_empty() -> None:
    """A single question with no multi-question markers returns []."""
    assert split_user_questions("How much PTO do I accrue per year?") == []


def test_split_empty_string_returns_empty() -> None:
    assert split_user_questions("") == []


def test_split_no_question_marks_returns_empty() -> None:
    """Declarative sentence without question marks cannot be split."""
    assert split_user_questions("Tell me about PTO and also the travel budget") == []


def test_split_three_question_marks() -> None:
    result = split_user_questions("What is PTO? What is the budget? Can I expense therapy?")
    assert len(result) == 3


def test_split_numbered_with_closing_paren() -> None:
    """Supports '1)' style numbering in addition to '1.'."""
    result = split_user_questions("1) How much PTO? 2) What is the budget?")
    assert len(result) == 2


# ── route_question_fast ──────────────────────────────────────────────


def test_fast_router_detects_departments() -> None:
    assert route_question_fast("How much PTO do I accrue?") == "hr"
    assert route_question_fast("What is the hotel budget for travel?") == "finance"
    assert route_question_fast("Can I expense a wellness benefit?") == "both"


def test_fast_router_returns_none_for_ambiguous() -> None:
    """No HR or Finance terms → should return None (fall through to LLM)."""
    assert route_question_fast("Where is the office located?") is None


def test_fast_router_substring_matching() -> None:
    """'ap' (FINANCE_TERM) matches inside 'happens' via substring search.

    This documents existing behavior — substring matching is a known
    tradeoff for speed. If this becomes a problem, switch to word-boundary
    matching in _contains_any.
    """
    # 'happens' contains 'ap', which is a finance term
    assert route_question_fast("What happens if I submit it late?") == "finance"


def test_fast_router_is_case_insensitive() -> None:
    assert route_question_fast("What is my PTO balance?") == "hr"
    assert route_question_fast("WHAT IS MY PTO BALANCE?") == "hr"


def test_fast_router_detects_hr_terms() -> None:
    assert route_question_fast("What is the onboarding process?") == "hr"
    assert route_question_fast("Tell me about parental leave") == "hr"
    assert route_question_fast("Is there a wellness program?") == "hr"


def test_fast_router_detects_finance_terms() -> None:
    assert route_question_fast("How does procurement work?") == "finance"
    assert route_question_fast("What is the per diem rate?") == "finance"
    assert route_question_fast("Can I get reimbursement for this?") == "finance"


# ── is_vague_subquestion ─────────────────────────────────────────────


def test_vague_subquestion_requires_clarification() -> None:
    assert is_vague_subquestion("How much do I get?")


def test_vague_known_patterns() -> None:
    """All hardcoded VAGUE_PATTERNS should be detected."""
    assert is_vague_subquestion("What is the deadline?")
    assert is_vague_subquestion("Do I need approval?")
    assert is_vague_subquestion("What happens if I submit it late?")


def test_vague_with_trailing_whitespace() -> None:
    assert is_vague_subquestion("  How much do I get?  ")


def test_not_vague_when_hr_term_present() -> None:
    assert not is_vague_subquestion("How much PTO do I get?")


def test_not_vague_when_finance_term_present() -> None:
    assert not is_vague_subquestion("How much reimbursement do I get?")


def test_no_department_terms_is_vague() -> None:
    """A question with no HR or Finance terms is treated as vague."""
    assert is_vague_subquestion("Where is the office located?")


def test_approves_not_vague_due_to_substring() -> None:
    """'approves' contains 'ap' (FINANCE_TERM), so it's not vague."""
    assert not is_vague_subquestion("Who approves this request?")


# ── split_department_questions_fast ───────────────────────────────────


def test_fast_split_separates_hr_and_finance() -> None:
    result = split_department_questions_fast(
        "What is my PTO accrual? What is the hotel budget for NYC?"
    )
    assert result is not None
    assert "pto" in result.hr_question.lower()
    assert "hotel" in result.finance_question.lower()


def test_fast_split_returns_none_for_single_department() -> None:
    """When all chunks belong to one department, fast split can't separate."""
    result = split_department_questions_fast("What is my PTO accrual?")
    assert result is None


def test_fast_split_returns_none_for_no_terms() -> None:
    """When no department terms are found, returns None (falls to LLM)."""
    result = split_department_questions_fast("What happens next? Who decides?")
    assert result is None
