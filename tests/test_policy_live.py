"""Destructive proof against an explicitly supplied disposable Neo4j database.

BOSS_POLICY_TEST_URI=bolt://127.0.0.1:17687 uv run pytest tests/test_policy_live.py -v
Never supply a production/shared database. Inference and embeddings are synthetic;
Neo4j transactions, HTTP, cache, ingestion and separate API processes are real.
"""
from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import socket
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

from app import policy
from app.api import main
from app.config import settings
from app.retrieval.rag import EMBEDDING_DIMENSIONS
from scripts import ingest
from tests.helpers import auth_headers

pytestmark = pytest.mark.skipif(not os.getenv("BOSS_POLICY_TEST_URI"), reason="requires disposable Neo4j")


class Embedder:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [[1.] + [0.] * (EMBEDDING_DIMENSIONS - 1) for _ in texts]

    def embed_query(self, question):
        return [1.] + [0.] * (EMBEDDING_DIMENSIONS - 1)


@pytest.fixture
def live_graph(monkeypatch):
    uri = os.environ["BOSS_POLICY_TEST_URI"]
    monkeypatch.setenv("NEO4J_URI", uri)
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", os.getenv("BOSS_POLICY_TEST_PASSWORD", "test-password"))
    monkeypatch.setattr(settings, "neo4j_uri", uri)
    from pydantic import SecretStr
    monkeypatch.setattr(settings, "neo4j_password", SecretStr(os.environ["NEO4J_PASSWORD"]))
    driver = GraphDatabase.driver(uri, auth=("neo4j", os.environ["NEO4J_PASSWORD"]))
    monkeypatch.setattr(policy, "get_driver", lambda: driver)
    from app.api import policy as transport
    monkeypatch.setattr(transport, "gate", policy.gate)
    monkeypatch.setattr(main, "gate", policy.gate)
    graph = ingest.connect()
    graph.query("MATCH (n) DETACH DELETE n")
    yield graph
    graph.query("MATCH (n) DETACH DELETE n")
    graph._driver.close()
    driver.close()


def corpus():
    return {"hr": [Document(page_content="Allowance is $900.", metadata={
        "source": "HR-OLD", "title": "Allowance", "department": "HR",
        "acl_groups": ["all-employees"], "effective_date": "2026-01-01",
    })], "finance": []}


def sync(graph, docs, embedder=None, **kwargs):
    return ingest.ingest_corpus(graph, docs, RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0),
                                embedder=embedder or Embedder(), **kwargs)


def test_atomic_close_registration_and_crash_recovery(live_graph):
    sync(live_graph, corpus())
    gate = policy.gate
    barrier = threading.Barrier(13)

    def admit():
        barrier.wait()
        try:
            return gate.admit()[0]
        except policy.PolicyUnavailable:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        pending = [pool.submit(admit) for _ in range(12)]
        barrier.wait()
        owner = gate.close()
        admitted = {ticket for f in pending if (ticket := f.result()) is not None}
    assert set(gate.snapshot()["active"]) == admitted
    with pytest.raises(policy.PolicyUnavailable):
        gate.admit()
    for ticket in admitted:
        gate.release(ticket)
    assert gate.drained(owner)
    gate.publish(owner)

    # Lost API process leaves a real durable registration, not a stale local count.
    ticket, _ = gate.admit()
    owner = gate.close()
    with pytest.raises(policy.PolicyUnavailable, match="timed out"):
        gate.drain(owner, timeout=0)
    assert ticket in policy.PolicyGate().snapshot()["active"]
    with pytest.raises(policy.PolicyUnavailable):
        policy.PolicyGate().admit()
    with pytest.raises(policy.PolicyUnavailable):
        gate.close("wrong-cycle")
    # Explicit stopped-process recovery works after an ingester restart too.
    recovered = policy.PolicyGate().close(owner)
    assert gate.drained(recovered)
    gate.fail(recovered)
    sync(live_graph, corpus())
    assert gate.snapshot()["open"] is True


def test_metadata_embedding_preservation_and_verification_failure(live_graph, monkeypatch):
    docs, embedder = corpus(), Embedder()
    sync(live_graph, docs, embedder)
    original = live_graph.query("MATCH (c:HRDocument) RETURN c.embedding AS vector")[0]["vector"]
    docs["hr"][0].metadata["acl_groups"] = ["executives"]
    stats = sync(live_graph, docs, embedder)
    assert stats["metadata_changed"] == 1 and embedder.calls == 1
    row = live_graph.query("MATCH (c:HRDocument) RETURN properties(c) AS p")[0]["p"]
    assert row["acl_groups"] == ["executives"] and row["embedding"] == original
    docs["hr"][0].metadata["superseded_by"] = "HR-NEW"
    stats = sync(live_graph, docs, embedder)
    row = live_graph.query("MATCH (c:HRDocument) RETURN properties(c) AS p")[0]["p"]
    assert stats["metadata_changed"] == 1 and row["superseded_by"] == "HR-NEW"
    assert row["embedding"] == original and embedder.calls == 1
    docs["hr"][0].metadata.pop("superseded_by")
    sync(live_graph, docs, embedder)
    assert live_graph.query("MATCH (c:HRDocument) RETURN c.superseded_by AS s")[0]["s"] is None

    verify = ingest.verify_corpus
    def corrupt_before_verify(graph, chunks):
        graph.query("MATCH (c:HRDocument) SET c.acl_groups = ['all-employees']")
        verify(graph, chunks)
    monkeypatch.setattr(ingest, "verify_corpus", corrupt_before_verify)
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        sync(live_graph, docs, embedder)
    assert policy.gate.snapshot()["open"] is False
    monkeypatch.setattr(ingest, "verify_corpus", verify)
    sync(live_graph, docs, embedder)
    assert policy.gate.snapshot()["open"] is True


def test_partial_write_failure_reset_and_restart_stay_closed(live_graph):
    docs = corpus()
    sync(live_graph, docs)
    docs["hr"][0].metadata["acl_groups"] = ["executives"]
    docs["finance"] = [Document(page_content="Finance rule", metadata={"source": "FIN-1", "acl_groups": ["all-employees"]})]

    class BrokenEmbedding:
        def embed_documents(self, texts):
            raise OSError("provider unavailable")

    with pytest.raises(OSError):
        sync(live_graph, docs, BrokenEmbedding())
    assert live_graph.query("MATCH (c:HRDocument) RETURN c.acl_groups AS acl")[0]["acl"] == ["executives"]
    assert policy.PolicyGate().snapshot()["open"] is False
    with pytest.raises(policy.PolicyUnavailable):
        policy.PolicyGate().admit()
    sync(live_graph, docs)
    with pytest.raises(OSError):
        sync(live_graph, docs, BrokenEmbedding(), reset=True)
    state = policy.PolicyGate().snapshot()
    assert state["open"] is False and not state["active"]
    sync(live_graph, docs)
    assert policy.gate.snapshot()["revision"] > state["revision"]


def worker(port, entered, finish):
    """Real API process; replace only the external embedding/model boundary."""
    import uvicorn
    from app.agents import router
    from app.cache import get_cache
    from app.db import get_driver
    settings.enable_semantic_cache = True
    get_cache().invalidate_all()
    router.get_embedder = Embedder

    def generate(question, department, mode, groups):
        if question == "slow":
            entered.set()
            if not finish.wait(30):
                raise RuntimeError("test generation deadline")
        with get_driver().session(database=settings.neo4j_database) as session:
            rows = session.run(
                "MATCH (c:HRDocument) WHERE c.superseded_by IS NULL "
                "AND any(g IN c.acl_groups WHERE g IN $groups) "
                "RETURN c.text AS text, c.source AS source ORDER BY source",
                groups=sorted(groups),
            ).data()
        return {"answer": " ".join(r["text"] for r in rows) or "NO ACCESS",
                "sources": [{"source": r["source"]} for r in rows], "department_routed": "hr",
                "contexts": [r["text"] for r in rows], "abstained": not rows}

    router._answer_question_uncached = generate

    def stream(question, department, mode, groups):
        yield "meta", {}
        answer = router.answer_question(question, department, mode=mode, groups=groups)
        yield "token", answer["answer"]
        yield "done", answer["answer"]

    main.stream_question = stream
    uvicorn.run(main.app, host="127.0.0.1", port=port, log_level="error")


@contextmanager
def running_worker():
    ctx = multiprocessing.get_context("spawn")
    entered, finish = ctx.Event(), ctx.Event()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = ctx.Process(target=worker, args=(port, entered, finish))
    process.start()
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                if httpx.get(url + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline or not process.is_alive():
                raise RuntimeError("API worker did not start")
            time.sleep(0.05)
        yield process, url, entered, finish
    finally:
        # Never notify a condition after SIGKILL: its dead waiter cannot
        # acknowledge multiprocessing.Condition.notify_all().
        if process.is_alive():
            finish.set()
            process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(5)


def ask(url, endpoint="/ask", question="allowance"):
    return httpx.post(url + endpoint, headers=auth_headers(), json={"question": question, "department": "hr"}, timeout=20)


@pytest.mark.parametrize("endpoint", ["/ask", "/ask/stream"])
def test_multiworker_http_drain_revocation_supersession_cache_and_crash(live_graph, endpoint):
    docs = corpus()
    sync(live_graph, docs)
    with running_worker() as first, running_worker() as second:
        for current in (first, second):
            assert "$900" in ask(current[1], endpoint).text
        docs["hr"][0].metadata["acl_groups"] = ["executives"]
        sync(live_graph, docs)
        for current in (first, second):
            response = ask(current[1], endpoint)
            assert "NO ACCESS" in response.text and "$900" not in response.text
        docs["hr"][0].metadata["acl_groups"] = ["all-employees"]
        docs["hr"][0].metadata["superseded_by"] = "HR-NEW"
        docs["hr"].append(Document(page_content="Allowance is $950.", metadata={
            "source": "HR-NEW", "acl_groups": ["all-employees"], "department": "HR",
        }))
        sync(live_graph, docs)
        for current in (first, second):
            response = ask(current[1], endpoint)
            assert "$950" in response.text and "$900" not in response.text

        # Fresh worker caches ensure the next request actually enters generation.
    with running_worker() as first, running_worker() as second:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            active = pool.submit(ask, first[1], endpoint, "slow")
            assert first[2].wait(10)
            owner = policy.gate.close()
            assert not policy.gate.drained(owner)
            for current in (first, second):
                assert ask(current[1], "/ask").status_code == 503
                assert ask(current[1], "/ask/stream").status_code == 503
            first[3].set()
            assert active.result().status_code == 200
            policy.gate.drain(owner, 5)
            policy.gate.fail(owner)
        sync(live_graph, docs)

        # A real TCP disconnect must not free the generation registration.
        first[2].clear()
        first[3].clear()
        body = json.dumps({"question": "slow", "department": "hr"}).encode()
        authorization = auth_headers()["Authorization"]
        with socket.create_connection(("127.0.0.1", int(first[1].rsplit(":", 1)[1])), timeout=5) as client:
            client.sendall((
                f"POST {endpoint} HTTP/1.1\r\nHost: localhost\r\n"
                f"Authorization: {authorization}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n\r\n"
            ).encode() + body)
            assert first[2].wait(10)
        owner = policy.gate.close()
        assert not policy.gate.drained(owner), "TCP disconnect abandoned active generation"
        assert ask(second[1], endpoint).status_code == 503
        first[3].set()
        policy.gate.drain(owner, 5)
        policy.gate.fail(owner)
        sync(live_graph, docs)

    # A killed API leaves a recoverable, fail-closed durable registration.
    with running_worker() as crashed:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            active = pool.submit(ask, crashed[1], endpoint, "slow")
            assert crashed[2].wait(10)
            crashed[0].kill()
            crashed[0].join(5)
            try:
                active.result()
            except httpx.HTTPError:
                pass
        owner = policy.gate.close()
        with pytest.raises(policy.PolicyUnavailable, match="timed out"):
            policy.gate.drain(owner, 0)
        assert policy.gate.snapshot()["active"]
    # All API/ingestion workers from that cycle are now stopped.
    sync(live_graph, docs, recover_stopped_workers=owner)
    with running_worker() as restarted:
        response = ask(restarted[1], endpoint)
        assert response.status_code == 200 and "$950" in response.text and "$900" not in response.text


def test_state_read_failure_requires_verified_recovery(live_graph, monkeypatch):
    sync(live_graph, corpus())
    driver_factory = policy.get_driver

    def unavailable():
        raise OSError("database state cannot be read")

    with monkeypatch.context() as fault:
        fault.setattr(policy, "get_driver", unavailable)
        from fastapi.testclient import TestClient
        for endpoint in ("/ask", "/ask/stream"):
            response = TestClient(main.app).post(endpoint, headers=auth_headers(), json={"question": "PTO?"})
            assert response.status_code == 503
    assert policy.get_driver is driver_factory
    # Reconnected storage is not permission to serve the unverified corpus.
    with pytest.raises(policy.PolicyUnavailable):
        policy.gate.admit()
    assert policy.gate.snapshot()["open"] is False
    sync(live_graph, corpus())
    ticket, epoch = policy.gate.admit()
    policy.gate.release(ticket)
    # State loss plus recreation must never reuse an old cache publication key.
    live_graph.query("MATCH (g:PolicyGate) DETACH DELETE g")
    with pytest.raises(policy.PolicyUnavailable):
        policy.gate.admit()
    sync(live_graph, corpus())
    ticket, replacement_epoch = policy.gate.admit()
    assert replacement_epoch != epoch
    policy.gate.release(ticket)
