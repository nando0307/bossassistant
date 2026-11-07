"""Real OpenAI HTTP adapter and Boss public contracts, with synthetic evidence only."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from pydantic import SecretStr

from app.agents import router
from app.api import main
from app.cache import SemanticCache
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
                finish = state["finish"]
                if body.get("response_format"):
                    text = json.dumps(state.get("structured", {"department": state.get("route", "finance")}))
                    finish = "stop"
                self.wfile.write(json.dumps({"id": "test", "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(settings, "generation_base_url", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setattr(settings, "generation_api_key", SecretStr("synthetic-service-key"))
    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: [Document(page_content="Synthetic policy grants 15 days.", metadata={"source": "HR-SYNTHETIC", "acl_groups": ["all-employees"]})])
    rag.get_llm.cache_clear()
    try:
        yield state
        for body in state["requests"]:
            assert "tools" not in body
            assert "tool_choice" not in body
    finally:
        rag.get_llm.cache_clear()
        server.shutdown()
        server.server_close()
        thread.join()


def test_nightly_eval_supplies_gateway_configuration() -> None:
    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github/workflows/nightly-eval.yml").read_text())
    env = workflow["jobs"]["eval"]["env"]
    assert env["GENERATION_BASE_URL"] == "${{ vars.GENERATION_BASE_URL }}"
    assert env["GENERATION_API_KEY"] == "${{ secrets.GENERATION_API_KEY }}"


def test_mixed_generation_paths_reuse_model_clients(gateway: Any) -> None:
    clients = (
        rag.get_llm_for_mode("fast"), rag.get_llm_for_mode("deep"),
        graphrag.get_map_llm(), graphrag.get_reduce_llm(),
    )
    for _ in range(2):
        assert router.route_question("Synthetic zephyr procedure?") == "finance"
        router.split_department_questions("PTO and travel allowances?")
        router.rewrite_followup("What about contractors?", [{"role": "user", "content": "PTO?"}])
        rag.generate_queries("PTO?")
        current = (
            rag.get_llm_for_mode("fast"), rag.get_llm_for_mode("deep"),
            graphrag.get_map_llm(), graphrag.get_reduce_llm(),
        )
        assert all(actual is original for actual, original in zip(current, clients, strict=True))


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


@pytest.mark.parametrize("min_hits", [3, 4])
def test_graph_counts_all_support_before_bounding_context(
    gateway: Any, monkeypatch: Any, min_hits: int,
) -> None:
    """R2's worked example: only C has three entity hits, even in third position."""
    rows = [
        {"chunks": [{"source": source, "text": f"chunk {index} of {source}"} for source in sources]}
        for index, sources in enumerate((("A", "B", "C"), ("D", "E", "C"), ("F", "G", "C")))
    ]
    rows[0]["chunks"].extend([rows[0]["chunks"][0]] * 3)
    monkeypatch.setattr(graphrag, "connect", lambda: SimpleNamespace(query=lambda *a: rows))
    result = graphrag.local_search("approval", min_entity_hits=min_hits, groups=frozenset({"all-employees"}))
    assert "C" in result["sources"]
    assert len(result["sources"]) == (1 if min_hits == 3 else 4)
    assert [text for text in result["contexts"] if text.startswith("[C]")] == [
        "[C]\nchunk 0 of C", "[C]\nchunk 1 of C",
    ]
    assert len(result["contexts"]) == len(set(result["contexts"]))
    if min_hits == 3:
        assert result["contexts"] == ["[C]\nchunk 0 of C", "[C]\nchunk 1 of C"]


@pytest.mark.parametrize("finish", [None, "length", "content_filter", "stop"])
@pytest.mark.parametrize("endpoint", ["/ask", "/ask/stream"])
@pytest.mark.parametrize("payload", [
    {"question": "PTO?", "department": "hr"},
    {"question": "Travel?", "department": "finance", "mode": "deep"},
    {"question": "PTO?"},
    {"question": "PTO and travel allowances?"},
    {"question": "PTO? Travel?"},
    {"question": "PTO?", "mode": "graph"},
])
def test_final_answers_require_normal_completion_before_success_or_cache(
    gateway: Any, monkeypatch: Any, finish: str | None, endpoint: str, payload: dict[str, str],
) -> None:
    """The public completion contract rejects partial answers on every answer route."""
    gateway["finish"] = finish
    cache = SemanticCache()
    monkeypatch.setattr(settings, "enable_semantic_cache", True)
    monkeypatch.setattr(router, "get_cache", lambda: cache)
    monkeypatch.setattr(router, "get_embedder", lambda: SimpleNamespace(embed_query=lambda q: [1.0, 0.0]))
    rows = [{"chunks": [{"source": "HR-SYNTHETIC", "text": "Synthetic policy grants 15 days."}]}]
    monkeypatch.setattr(graphrag, "connect", lambda: SimpleNamespace(query=lambda *a: rows))
    response = TestClient(main.app).post(endpoint, headers=auth_headers(), json=payload)
    if finish != "stop":
        assert cache.size() == 0
        if endpoint == "/ask":
            assert response.status_code == 503
        else:
            result = events(response)
            assert all(name != "done" for name, _ in result)
            assert result[-1][0] == "error" and result[-1][1]["retryable"] is True
    else:
        assert response.status_code == 200
        if endpoint == "/ask":
            assert gateway["text"] in response.json()["answer"]
            assert cache.size() == 1
        else:
            result = events(response)
            assert result[-1][0] == "done" and gateway["text"] in result[-1][1]


@pytest.mark.parametrize("finish", [None, "length", "content_filter", "stop"])
def test_global_graph_reduce_requires_normal_completion(
    gateway: Any, monkeypatch: Any, finish: str | None,
) -> None:
    """Offline global reduction must not return an incomplete final answer either."""
    gateway.update(finish=finish, structured={"points": [{"point": "15 days", "score": 100}]})
    monkeypatch.setattr(graphrag, "connect", lambda: SimpleNamespace(query=lambda *a: [{"source": "HR-SYNTHETIC"}]))
    monkeypatch.setattr(graphrag, "rank_communities", lambda *a: [{"id": "c1", "title": "PTO", "summary": "15 days"}])
    if finish != "stop":
        with pytest.raises(rag.GenerationInterruptedError):
            graphrag.global_search("PTO?")
    else:
        result = graphrag.global_search("PTO?")
        assert result["answer"] == gateway["text"]
        assert result["sources"] == ["HR-SYNTHETIC"]
    assert [request["model"] for request in gateway["requests"]] == ["boss-graph-map", "boss-graph-reduce"]
