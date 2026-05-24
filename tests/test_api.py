from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.api import main


def test_ask_defaults_to_fast_mode(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_answer_question(question: str, department: str | None = None, mode: str = "fast") -> dict[str, Any]:
        captured.update({"question": question, "department": department, "mode": mode})
        return {"answer": "ok", "sources": [], "department_routed": "hr"}

    monkeypatch.setattr(main, "answer_question", fake_answer_question)
    response = TestClient(main.app).post("/ask", json={"question": "How much PTO do I accrue?"})

    assert response.status_code == 200
    assert captured == {
        "question": "How much PTO do I accrue?",
        "department": None,
        "mode": "fast",
    }


def test_ask_accepts_deep_mode(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_answer_question(question: str, department: str | None = None, mode: str = "fast") -> dict[str, Any]:
        captured.update({"question": question, "department": department, "mode": mode})
        return {"answer": "ok", "sources": [], "department_routed": "finance"}

    monkeypatch.setattr(main, "answer_question", fake_answer_question)
    response = TestClient(main.app).post(
        "/ask",
        json={
            "question": "What is the hotel budget?",
            "department": "finance",
            "mode": "deep",
        },
    )

    assert response.status_code == 200
    assert captured == {
        "question": "What is the hotel budget?",
        "department": "finance",
        "mode": "deep",
    }
