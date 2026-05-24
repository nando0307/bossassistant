from __future__ import annotations

from app.agents.router import is_vague_subquestion, route_question_fast, split_user_questions


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


def test_vague_subquestion_requires_clarification() -> None:
    assert is_vague_subquestion("How much do I get?")


def test_fast_router_detects_departments() -> None:
    assert route_question_fast("How much PTO do I accrue?") == "hr"
    assert route_question_fast("What is the hotel budget for travel?") == "finance"
    assert route_question_fast("Can I expense a wellness benefit?") == "both"
