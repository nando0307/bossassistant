"""ASGI admission covers generation AND the last response byte, including SSE.

Registration happens only after the request body is parsed and the caller is
authenticated, so an unauthenticated or stalled request can never hold a
durable registration. A disconnected client does not cancel policy work: the
app runs to completion in a shielded task, discarding output after disconnect,
so synchronous model threads finish before the registration is released.
"""
from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar, Token

from anyio import to_thread
from fastapi import Depends, HTTPException, status
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.auth import Principal, get_principal
from app.policy import gate, policy_revision

log = logging.getLogger(__name__)

_UNAVAILABLE = "policy maintenance or shared state unavailable"


class Admission:
    """One request's registration with the shared gate."""

    def __init__(self) -> None:
        self.ticket: str | None = None
        self.revision: str | None = None
        self._token: Token[str | None] | None = None

    def activate(self, ticket: str, revision: str) -> None:
        self.ticket = ticket
        self.revision = revision
        self._token = policy_revision.set(revision)

    def check(self) -> None:
        if self.ticket is None or self.revision is None:
            raise RuntimeError("Policy check without a registration")
        gate.check(self.ticket, self.revision)

    def release(self) -> None:
        ticket, self.ticket = self.ticket, None
        if ticket is not None:
            gate.release(ticket)

    def reset_context(self) -> None:
        if self._token is not None:
            policy_revision.reset(self._token)
            self._token = None


_current: ContextVar[Admission | None] = ContextVar("policy_admission", default=None)


async def require_policy_admission(
    _principal: Principal = Depends(get_principal),
) -> None:
    """Register after body parsing and authentication, before any policy work."""
    admission = _current.get()
    if admission is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_UNAVAILABLE,
            headers={"Retry-After": "30"},
        )
    try:
        ticket, revision = await to_thread.run_sync(gate.admit)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_UNAVAILABLE,
            headers={"Retry-After": "30"},
        ) from None
    admission.activate(ticket, revision)


class PolicyAdmission:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.tasks: set[asyncio.Task[None]] = set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"].rstrip("/") not in {"/ask", "/ask/stream"}:
            await self.app(scope, receive, send)
            return
        task = asyncio.create_task(self._serve(scope, receive, send))
        self.tasks.add(task)
        task.add_done_callback(self._finished)
        # Client/server request cancellation must not abandon a model thread.
        await asyncio.shield(task)

    def _finished(self, task: asyncio.Task[None]) -> None:
        self.tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("Policy response failed", exc_info=task.exception())

    async def _serve(self, scope: Scope, receive: Receive, send: Send) -> None:
        unavailable = JSONResponse(
            {"detail": _UNAVAILABLE},
            status_code=503,
            headers={"Retry-After": "30"},
        )
        admission = Admission()
        token = _current.set(admission)
        disconnected = False
        body_complete = False
        started = False
        completed = False

        async def keep_receiving() -> Message:
            nonlocal disconnected, body_complete
            while True:
                message = await receive()
                if message["type"] == "http.request":
                    if not message.get("more_body", False):
                        body_complete = True
                    return message
                if message["type"] != "http.disconnect":
                    continue
                disconnected = True
                if not body_complete:
                    # Nothing was generated yet; let the app abort so this
                    # registration is released instead of pinned forever.
                    return message
                # StreamingResponse's disconnect listener waits until generation
                # finishes; its normal task-group exit cancels this listener.
                await asyncio.Future()

        async def guarded_send(message: Message) -> None:
            nonlocal disconnected, started
            if disconnected:
                return
            if admission.ticket is not None:
                try:
                    await to_thread.run_sync(admission.check)
                except Exception:
                    disconnected = True
                    if not started:
                        await unavailable(scope, receive, send)
                    return
            try:
                await send(message)
                started = True
            except OSError:
                disconnected = True

        try:
            try:
                await self.app(scope, keep_receiving, guarded_send)
            except Exception:
                # Normal exception unwinding has joined synchronous work. A
                # BaseException cancellation is different: leave a durable
                # registration for stopped-worker recovery, never claim drain.
                completed = True
                raise
            completed = True
        finally:
            admission.reset_context()
            _current.reset(token)
            if completed and admission.ticket is not None:
                ticket = admission.ticket
                try:
                    await to_thread.run_sync(admission.release)
                except Exception:
                    # An uncertain release must not turn into false drain.
                    log.exception("Policy registration retained after release failure: %s", ticket)
