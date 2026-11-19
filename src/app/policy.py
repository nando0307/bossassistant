"""Shared maintenance admission in the existing Neo4j database.

The singleton's write lock serializes registration, closure and publication,
not generation. Registrations are durable, not expiring leases: a partitioned
worker can still be generating. Lost registrations require stopped-worker
recovery, never a timeout-based assertion that policy work has ended.
"""
from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, TypeVar
from uuid import uuid4

from app.config import settings
from app.db import get_driver

policy_revision: ContextVar[str | None] = ContextVar("policy_revision", default=None)
T = TypeVar("T")


class PolicyUnavailable(RuntimeError):
    pass


class PolicyGate:
    def __init__(self) -> None:
        self._fault_lock = threading.Lock()
        self._uncertain = False

    def initialize(self) -> None:
        """Only maintenance initializes state. Missing state never admits traffic."""
        with get_driver().session(database=settings.neo4j_database) as session:
            session.run(
                "CREATE CONSTRAINT policy_gate_key IF NOT EXISTS "
                "FOR (g:PolicyGate) REQUIRE g.key IS UNIQUE"
            ).consume()
            session.run(
                "MERGE (g:PolicyGate {key: 'corpus'}) "
                "ON CREATE SET g.open = false, g.revision = 0, g.active = [], g.serial = 0"
            ).consume()

    @staticmethod
    def _parse(record: Any) -> dict[str, Any]:
        if record is None:
            raise PolicyUnavailable("Policy state is missing; run verified ingestion")
        state = dict(record["state"])
        if not isinstance(state.get("active"), list) or not isinstance(state.get("revision"), int):
            raise PolicyUnavailable("Invalid policy state")
        return state

    def _clear_uncertainty(self, session: Any) -> None:
        # If a state operation failed, never turn restored connectivity into
        # implicit recovery. Persist closure before doing any more work. Only
        # verified ingestion can subsequently reopen it.
        with self._fault_lock:
            if not self._uncertain:
                return
            session.run(
                "MATCH (g:PolicyGate {key: 'corpus'}) SET g.open = false"
            ).consume()
            self._uncertain = False

    def _read(self) -> dict[str, Any]:
        """Read-only: the per-response fence must not serialize on the write lock."""
        def read(tx: Any) -> dict[str, Any]:
            return self._parse(
                tx.run(
                    "MATCH (g:PolicyGate {key: 'corpus'}) RETURN properties(g) AS state"
                ).single()
            )

        try:
            with get_driver().session(database=settings.neo4j_database) as session:
                self._clear_uncertainty(session)
                return session.execute_read(read)
        except PolicyUnavailable:
            raise
        except Exception:
            with self._fault_lock:
                self._uncertain = True
            raise

    def _write(self, change: Callable[[Any, dict[str, Any]], T]) -> T:
        def transaction(tx: Any) -> T:
            # Dependent update takes the lock BEFORE reading state. Neo4j can
            # deadlock under contention; execute_write retries rolled-back work.
            # All mutations below are idempotent under an ambiguous commit too.
            record = tx.run(
                "MATCH (g:PolicyGate {key: 'corpus'}) "
                "SET g.serial = g.serial + 1 RETURN properties(g) AS state"
            ).single()
            return change(tx, self._parse(record))

        try:
            with get_driver().session(database=settings.neo4j_database) as session:
                self._clear_uncertainty(session)
                return session.execute_write(transaction)
        except PolicyUnavailable:
            raise
        except Exception:
            with self._fault_lock:
                self._uncertain = True
            raise

    def snapshot(self) -> dict[str, Any]:
        return self._read()

    def admit(self) -> tuple[str, str]:
        ticket = f"{socket.gethostname()}:{os.getpid()}:{uuid4()}"

        def register(tx: Any, state: dict[str, Any]) -> tuple[str, str]:
            if state.get("open") is not True or state.get("owner"):
                raise PolicyUnavailable("Policy maintenance in progress")
            if not isinstance(state.get("published_by"), str):
                raise PolicyUnavailable("Missing verified publication identity")
            if ticket not in state["active"]:
                tx.run(
                    "MATCH (g:PolicyGate {key: 'corpus'}) SET g.active = g.active + $ticket",
                    ticket=[ticket],
                ).consume()
            # A UUID cannot collide with an old process cache even if a lost
            # gate is reinitialized and its numeric revision starts over.
            return ticket, state["published_by"]

        return self._write(register)

    def check(self, ticket: str, revision: str) -> None:
        """Fence output from a registration invalidated by stopped-worker recovery."""
        state = self._read()
        if ticket not in state["active"] or revision != state.get("published_by"):
            raise PolicyUnavailable("Policy registration is no longer valid")

    def release(self, ticket: str) -> None:
        def remove(tx: Any, state: dict[str, Any]) -> None:
            tx.run(
                "MATCH (g:PolicyGate {key: 'corpus'}) "
                "SET g.active = [id IN g.active WHERE id <> $ticket]",
                ticket=ticket,
            ).consume()
        self._write(remove)

    def close(self, recover_stopped_workers: str | None = None) -> str:
        """Recovery token attests ALL old API and ingestion processes are stopped."""
        self.initialize()
        owner = str(uuid4())

        def close_gate(tx: Any, state: dict[str, Any]) -> str:
            if state.get("owner") == owner:
                return owner
            if recover_stopped_workers is not None:
                if state.get("open") is not False or state.get("cycle") != recover_stopped_workers:
                    raise PolicyUnavailable("Recovery requires the current CLOSED maintenance cycle")
                tx.run("MATCH (g:PolicyGate {key: 'corpus'}) SET g.active = []").consume()
            elif state.get("owner"):
                raise PolicyUnavailable(f"Maintenance already owned by {state['owner']}")
            tx.run(
                "MATCH (g:PolicyGate {key: 'corpus'}) "
                "SET g.open = false, g.owner = $owner, g.cycle = $owner",
                owner=owner,
            ).consume()
            return owner

        return self._write(close_gate)

    def drained(self, owner: str) -> bool:
        def check(tx: Any, state: dict[str, Any]) -> bool:
            self._require_owner(state, owner)
            return not state["active"]
        return self._write(check)

    def drain(self, owner: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not self.drained(owner):
            if time.monotonic() >= deadline:
                raise PolicyUnavailable(
                    f"Drain timed out; serving stays CLOSED. Maintenance cycle: {owner}. "
                    "Inspect active registrations; never expire live work."
                )
            time.sleep(0.1)

    @staticmethod
    def _require_owner(state: dict[str, Any], owner: str) -> None:
        if state.get("open") is not False or state.get("owner") != owner:
            raise PolicyUnavailable("Maintenance ownership lost")

    def publish(self, owner: str) -> int:
        """Call only after complete corpus verification under this owner."""
        def publish(tx: Any, state: dict[str, Any]) -> int:
            if state.get("open") is True and state.get("published_by") == owner:
                return state["revision"]
            self._require_owner(state, owner)
            if state["active"]:
                raise PolicyUnavailable("Policy work is still active")
            revision = state["revision"] + 1
            tx.run(
                "MATCH (g:PolicyGate {key: 'corpus'}) "
                "SET g.revision = $revision, g.open = true, g.published_by = $owner REMOVE g.owner",
                revision=revision, owner=owner,
            ).consume()
            return revision
        return self._write(publish)

    def fail(self, owner: str) -> None:
        """A retry may repair a failed run, but failure never opens admission."""
        def fail(tx: Any, state: dict[str, Any]) -> None:
            if state.get("cycle") == owner and state.get("open") is False and not state.get("owner"):
                return
            self._require_owner(state, owner)
            tx.run("MATCH (g:PolicyGate {key: 'corpus'}) REMOVE g.owner").consume()
        self._write(fail)


gate = PolicyGate()
