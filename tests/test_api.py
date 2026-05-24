from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.api import main


# ── Mode handling ────────────────────────────────────────────────────


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


# ── Input validation ────────────────────────────────────────────────


def test_ask_rejects_empty_question() -> None:
    """Pydantic min_length=1 on question should reject empty strings."""
    client = TestClient(main.app)
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_rejects_missing_question() -> None:
    """question is required."""
    client = TestClient(main.app)
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_rejects_invalid_department() -> None:
    """department must be 'hr', 'finance', or null."""
    client = TestClient(main.app)
    response = client.post("/ask", json={"question": "test", "department": "legal"})
    assert response.status_code == 422


def test_ask_rejects_invalid_mode() -> None:
    """mode must be 'fast' or 'deep'."""
    client = TestClient(main.app)
    response = client.post("/ask", json={"question": "test", "mode": "turbo"})
    assert response.status_code == 422


# ── Probe endpoints ─────────────────────────────────────────────────


def test_health_returns_ok() -> None:
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "env" in data

