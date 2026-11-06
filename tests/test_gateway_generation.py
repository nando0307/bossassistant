"""Real OpenAI HTTP adapter and Boss public contracts, with synthetic evidence only."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from pydantic import SecretStr

from app.api import main
from app.config import settings
from app.retrieval import graphrag, rag
from tests.helpers import auth_headers


@pytest.fixture
def gateway(monkeypatch: Any) -> Any:
    state: dict[str, Any] = {"requests": [], "text": "Synthetic policy grants 15 days.", "finish": "stop", "status": 200}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(body)
            assert self.path == "/v1/chat/completions"
            assert self.headers["Authorization"] == "Bearer synthetic-service-key"
            self.send_response(state["status"])
            self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
            if state.get("disconnect"):
                self.send_header("Content-Length", "100000")
            self.end_headers()
            if state["status"] != 200:
                self.wfile.write(json.dumps({"error": {"message": "exhausted", "type": "server_error", "code": "providers_exhausted"}}).encode())
            elif body.get("stream"):
                def emit(value: Any) -> None:
                    self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
                emit({"id": "test", "choices": [{"index": 0, "delta": {"content": state["text"]}, "finish_reason": None}]})
                if state.get("error"):
                    emit({"error": {"message": "interrupted", "type": "server_error", "code": "stream_interrupted"}})
                elif state["finish"]:
                    emit({"id": "test", "choices": [{"index": 0, "delta": {}, "finish_reason": state["finish"]}]})
                    self.wfile.write(b"data: [DONE]\n\n")
            else:
                text = state["text"]
                if body.get("response_format"):
                    text = json.dumps({"department": state.get("route", "finance")})
                self.wfile.write(json.dumps({"id": "test", "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(settings, "generation_base_url", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setattr(settings, "generation_api_key", SecretStr("synthetic-service-key"))
    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: [Document(page_content="Synthetic policy grants 15 days.", metadata={"source": "HR-SYNTHETIC", "acl_groups": ["all-employees"]})])
    rag.get_llm.cache_clear()
    try:
        yield state
    finally:
        rag.get_llm.cache_clear()
        server.shutdown()
        server.server_close()
        thread.join()


def events(response: Any) -> list[tuple[str, Any]]:
    return [(frame.splitlines()[0][7:], json.loads(frame.splitlines()[1][6:])) for frame in response.text.strip().split("\n\n")]


@pytest.mark.parametrize("failure", ["eof", "error", "length", "disconnect"])
@pytest.mark.parametrize("text", ["Synthetic policy grants 15 days.", rag.ABSTAIN_MARKER])
def test_incomplete_stream_never_emits_done(gateway: Any, failure: str, text: str) -> None:
    gateway.update(text=text, finish=None if failure in {"eof", "disconnect"} else "length" if failure == "length" else "stop", error=failure == "error", disconnect=failure == "disconnect")
    response = TestClient(main.app).post("/ask/stream", headers=auth_headers(), json={"question": "PTO?", "department": "hr"})
    result = events(response)
    assert all(name != "done" for name, _ in result)
    assert result[-1][0] == "error" and result[-1][1]["retryable"] is True
    assert rag.ABSTAIN_MARKER not in response.text
    assert len(gateway["requests"]) == 1


@pytest.mark.parametrize("text", ["Synthetic policy grants 15 days.", rag.ABSTAIN_MARKER, "Yes"])
def test_complete_stream_preserves_answer_and_abstention(gateway: Any, text: str) -> None:
    gateway["text"] = text
    response = TestClient(main.app).post("/ask/stream", headers=auth_headers(), json={"question": "PTO?", "department": "hr"})
    result = events(response)
    expected = "Not covered in policy. I checked the following and none of them answer this question: HR-SYNTHETIC." if text == rag.ABSTAIN_MARKER else text
    assert result[-1] == ("done", expected)
    assert "".join(value for name, value in result if name == "token") == expected


def test_exhaustion_survives_followup_rewrite_without_retry(gateway: Any) -> None:
    gateway["status"] = 503
    response = TestClient(main.app).post("/ask", headers=auth_headers(), json={"question": "What about contractors?", "department": "hr", "history": [{"role": "user", "content": "PTO policy?"}]})
    assert response.status_code == 503
    assert len(gateway["requests"]) == 1


def test_structured_route_is_validated(gateway: Any) -> None:
    client = TestClient(main.app, raise_server_exceptions=False)
    body = {"question": "What is the synthetic zephyr procedure?"}
    response = client.post("/ask", headers=auth_headers(), json=body)
    assert response.status_code == 200
    assert response.json()["department_routed"] == "finance"
    gateway["route"] = "unapproved-department"
    response = client.post("/ask", headers=auth_headers(), json=body)
    assert response.status_code == 500
    assert len(gateway["requests"]) == 3  # Invalid routing never reaches answer generation.


def test_graph_uses_authorized_chunk_provenance_not_merged_descriptions(gateway: Any, monkeypatch: Any) -> None:
    class Graph:
        def query(self, statement: str, params: dict[str, Any]) -> list[dict[str, Any]]:
            assert params["groups"] == ["all-employees"]
            return [{"name": "RESTRICTED_NAME", "type": "RESTRICTED_TYPE", "description": "RESTRICTED_DESCRIPTION", "neighbours": ["RESTRICTED_RELATIONSHIP"], "chunks": [{"source": "HR-SYNTHETIC", "text": "Synthetic policy grants 15 days."}]}]

    monkeypatch.setattr(graphrag, "connect", Graph)
    response = TestClient(main.app).post("/ask", headers=auth_headers(), json={"question": "PTO?", "mode": "graph", "include_contexts": True})
    assert response.status_code == 200
    assert response.json()["sources"][0]["source"] == "HR-SYNTHETIC"
    assert response.json()["contexts"] == ["[HR-SYNTHETIC]\nSynthetic policy grants 15 days."]
    prompt = json.dumps(gateway["requests"][0]["messages"])
    assert "RESTRICTED_" not in prompt
    assert "Synthetic policy grants 15 days." in prompt
