"""Response lifetime tests: a disconnect is not evidence of completed policy work."""
from __future__ import annotations

import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.policy import policy_revision
from tests.helpers import auth_headers


@pytest.mark.parametrize("endpoint", ["/ask", "/ask/stream"])
@pytest.mark.parametrize("failure", ["closed", "unreachable"])
def test_admission_refuses_without_running_policy(endpoint, failure, api_policy_gate, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Policy work began without admission")
    monkeypatch.setattr(main, "answer_question", forbidden)
    monkeypatch.setattr(main, "stream_question", forbidden)
    if failure == "closed":
        api_policy_gate.open = False
    else:
        def unreachable():
            raise OSError("shared state unavailable")
        monkeypatch.setattr(api_policy_gate, "admit", unreachable)
    response = TestClient(main.app).post(endpoint, headers=auth_headers(), json={"question": "PTO?"})
    assert response.status_code == 503
    assert api_policy_gate.active == set()


def scope_for(endpoint):
    return {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
        "method": "POST", "scheme": "http", "path": endpoint, "raw_path": endpoint.encode(),
        "query_string": b"", "root_path": "", "http_version": "1.1",
        "server": ("test", 80), "client": ("test", 1),
        "headers": [(b"content-type", b"application/json"), *[(k.lower().encode(), v.encode()) for k, v in auth_headers().items()]],
    }


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("endpoint", ["/ask", "/ask/stream"])
@pytest.mark.parametrize("disconnect", ["receive", "cancel", "send"])
def test_disconnected_generation_remains_registered(endpoint, disconnect, api_policy_gate, monkeypatch):
    entered, finish, ended = threading.Event(), threading.Event(), threading.Event()

    def answer(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        ended.set()
        return {"answer": "old policy", "sources": [], "department_routed": "hr"}

    def stream(*args, **kwargs):
        yield "meta", {}
        entered.set()
        assert finish.wait(5)
        ended.set()
        yield "done", "old policy"

    monkeypatch.setattr(main, "answer_question", answer)
    monkeypatch.setattr(main, "stream_question", stream)

    async def scenario():
        body_sent = False
        disconnected = asyncio.Event()
        output = []

        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": json.dumps({"question": "PTO?"}).encode()}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if disconnect == "send" and disconnected.is_set():
                raise OSError("client gone")
            output.append(message)

        task = asyncio.create_task(main.app(scope_for(endpoint), receive, send))
        try:
            await until(entered.is_set)
            assert len(api_policy_gate.active) == 1
            disconnected.set()
            if disconnect == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            api_policy_gate.open = False
            await asyncio.sleep(0.02)
            assert len(api_policy_gate.active) == 1, "HTTP cancellation falsely reported a drained model"
            assert not ended.is_set()
            finish.set()
            await until(lambda: not api_policy_gate.active)
            assert ended.is_set()
            if not task.cancelled():
                await task
        finally:
            finish.set()
            if not task.done():
                await task

    asyncio.run(scenario())


@pytest.mark.parametrize("endpoint", ["/ask", "/ask/stream"])
def test_disconnect_before_body_completion_releases_registration(
    endpoint, api_policy_gate, monkeypatch
):
    """An unauthenticated client that hangs up mid-body must not pin a registration."""
    def forbidden(*args, **kwargs):
        pytest.fail("Policy work began without a complete request body")
    monkeypatch.setattr(main, "answer_question", forbidden)
    monkeypatch.setattr(main, "stream_question", forbidden)

    async def scenario():
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b'{"question":', "more_body": True}
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = scope_for(endpoint)
        scope["headers"] = [(b"content-type", b"application/json")]
        task = asyncio.create_task(main.app(scope, receive, send))
        try:
            await asyncio.wait_for(task, 5)
        except Exception:
            pass
        assert api_policy_gate.active == set(), "partial-body disconnect leaked a registration"

    asyncio.run(scenario())


@pytest.mark.parametrize("endpoint", ["/ask", "/ask/stream"])
def test_registration_covers_last_response_send(endpoint, api_policy_gate, monkeypatch):
    monkeypatch.setattr(main, "answer_question", lambda *a, **k: {
        "answer": "policy", "sources": [], "department_routed": "hr",
    })
    monkeypatch.setattr(main, "stream_question", lambda *a, **k: iter([("done", "policy")]))

    async def scenario():
        last_send, release = asyncio.Event(), asyncio.Event()
        body_sent = False

        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": b'{"question":"PTO?"}'}
            await asyncio.Future()

        async def send(message):
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                last_send.set()
                await release.wait()

        task = asyncio.create_task(main.app(scope_for(endpoint), receive, send))
        await asyncio.wait_for(last_send.wait(), 5)
        api_policy_gate.open = False
        assert len(api_policy_gate.active) == 1
        release.set()
        await asyncio.wait_for(task, 5)
        assert not api_policy_gate.active

    asyncio.run(scenario())


def test_revision_partition_cannot_hit_an_old_answer(monkeypatch):
    from app.agents import router
    from app.cache import SemanticCache

    cache = SemanticCache()
    monkeypatch.setattr(router, "get_cache", lambda: cache)
    monkeypatch.setattr(router.settings, "enable_semantic_cache", True)
    class Embedder:
        def embed_query(self, question):
            return [1., 0.]
    monkeypatch.setattr(router, "get_embedder", Embedder)
    state = {"answer": "revoked allowance"}
    monkeypatch.setattr(router, "_answer_question_uncached", lambda *a: dict(state))
    token = policy_revision.set("publication-one")
    try:
        assert router.answer_question("allowance", groups=frozenset({"all-employees"}))["answer"] == "revoked allowance"
        state["answer"] = "NO ACCESS"
        assert router.answer_question("allowance", groups=frozenset({"all-employees"}))["answer"] == "revoked allowance"
        policy_revision.set("publication-two")
        assert router.answer_question("allowance", groups=frozenset({"all-employees"}))["answer"] == "NO ACCESS"
    finally:
        policy_revision.reset(token)
