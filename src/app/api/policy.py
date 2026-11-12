"""ASGI admission covers generation AND the last response byte, including SSE.

A disconnected client does not cancel policy work. The app runs to completion
in a shielded task, discarding output after disconnect. Thus synchronous model
threads and their executor children finish before the registration is released.
"""
from __future__ import annotations

import asyncio
import logging

from anyio import to_thread
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.policy import gate, policy_revision
from app.cache import get_cache

log = logging.getLogger(__name__)


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
            {"detail": "policy maintenance or shared state unavailable"},
            status_code=503,
            headers={"Retry-After": "30"},
        )
        try:
            ticket, revision = await to_thread.run_sync(gate.admit)
        except Exception:
            await unavailable(scope, receive, send)
            return

        token = policy_revision.set(revision)
        disconnected = False
        started = False
        completed = False

        async def keep_receiving() -> Message:
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
                # StreamingResponse's disconnect listener waits until generation
                # finishes; its normal task-group exit cancels this listener.
                await asyncio.Future()
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal disconnected, started
            if disconnected:
                return
            try:
                await to_thread.run_sync(gate.check, ticket, revision)
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
                get_cache().activate_revision(revision)
                await self.app(scope, keep_receiving, guarded_send)
            except Exception:
                # Normal exception unwinding has joined synchronous work. A
                # BaseException cancellation is different: leave a durable
                # registration for stopped-worker recovery, never claim drain.
                completed = True
                raise
            completed = True
        finally:
            policy_revision.reset(token)
            if completed:
                try:
                    await to_thread.run_sync(gate.release, ticket)
                except Exception:
                    # An uncertain release must not turn into false drain.
                    log.exception("Policy registration retained after release failure: %s", ticket)
