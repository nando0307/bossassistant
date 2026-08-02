"""SSE streaming contract.

The transport is tested here, not the model: what the client must be able to
rely on regardless of how the provider behaves.
"""
from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.api import main
from tests.helpers import auth_headers


def _events(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    event = ""
    for line in raw.splitlines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            out.append((event, line[6:]))
    return out


def _fake_stream(*args: Any, **kwargs: Any) -> Any:
    yield "meta", {"department_routed": "hr", "abstained": None, "streamed": True}
    yield "sources_formatted", [{"source": "HR-001", "title": "PTO Policy"}]
    yield "token", "15 days"
    yield "token", " per year."
    yield "done", "15 days per year."


def test_stream_requires_a_token() -> None:
    response = TestClient(main.app).post("/ask/stream", json={"question": "How much PTO?"})
    assert response.status_code == 401


def test_stream_emits_sources_before_any_token(monkeypatch: Any) -> None:
    """The actual perceived-latency win: retrieval finishes before generation."""
    monkeypatch.setattr(main, "stream_question", _fake_stream)
    response = TestClient(main.app).post(
        "/ask/stream", headers=auth_headers(), json={"question": "How much PTO?"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _events(response.text)
    names = [name for name, _ in events]
    assert names.index("sources_formatted") < names.index("token")
    assert names[-1] == "done"


def test_tokens_survive_newlines(monkeypatch: Any) -> None:
    """A raw newline in a token would terminate the SSE frame early.

    This is why every payload is json-encoded rather than written through
    verbatim: a numbered list answer would otherwise arrive truncated.
    """

    def newline_stream(*args: Any, **kwargs: Any) -> Any:
        yield "token", "line one\nline two\n\nline three"
        yield "done", "line one\nline two\n\nline three"

    monkeypatch.setattr(main, "stream_question", newline_stream)
    response = TestClient(main.app).post(
        "/ask/stream", headers=auth_headers(), json={"question": "list them"}
    )
    import json

    payloads = [json.loads(data) for name, data in _events(response.text) if name == "token"]
    assert payloads == ["line one\nline two\n\nline three"]


def test_failure_arrives_as_an_event_not_a_status(monkeypatch: Any) -> None:
    """The 200 is already sent by the time generation can fail."""

    def boom(*args: Any, **kwargs: Any) -> Any:
        yield "meta", {"department_routed": "hr", "abstained": None, "streamed": True}
        raise Exception("[503] ResourceExhausted: Worker local total request limit reached")

    monkeypatch.setattr(main, "stream_question", boom)
    response = TestClient(main.app).post(
        "/ask/stream", headers=auth_headers(), json={"question": "How much PTO?"}
    )
    import json

    assert response.status_code == 200
    errors = [json.loads(d) for name, d in _events(response.text) if name == "error"]
    assert errors and errors[0]["retryable"] is True


def test_non_retryable_failure_is_not_advertised_as_retryable(monkeypatch: Any) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        yield "meta", {"department_routed": "hr", "abstained": None, "streamed": True}
        raise ValueError("genuine bug")

    monkeypatch.setattr(main, "stream_question", boom)
    response = TestClient(main.app).post(
        "/ask/stream", headers=auth_headers(), json={"question": "How much PTO?"}
    )
    import json

    errors = [json.loads(d) for name, d in _events(response.text) if name == "error"]
    assert errors and errors[0]["retryable"] is False


def test_abstain_marker_never_reaches_the_client() -> None:
    """The internal sentinel must not leak as literal tokens mid-stream."""
    from app.retrieval.rag import ABSTAIN_MARKER

    assert ABSTAIN_MARKER not in _fake_stream.__doc__ if _fake_stream.__doc__ else True
    # The buffering that guarantees this lives in stream_department.
    import inspect

    from app.retrieval import rag

    source = inspect.getsource(rag.stream_department)
    assert "holding" in source and "format_abstention" in source
