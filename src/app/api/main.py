"""FastAPI application entrypoint.

Exposes a minimal HTTP API for BossAssistant. Two probes:

- `/health` — liveness. "Is the process alive?" Cheap, no I/O.
- `/ready`  — readiness. "Are upstream deps reachable?" Hits Neo4j.

Container orchestrators use these for different purposes: liveness
decides whether to restart the container, readiness decides whether
to route traffic. Conflating them causes restart loops during
upstream blips.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.router import answer_question, stream_question
from app.api.policy import PolicyAdmission
from app.auth import DEMO_PERSONAS, Principal, get_principal, issue_demo_token
from app.config import settings
from app.policy import gate
from app.retrieval.rag import RetrievalMode, retryable_generation_error

app = FastAPI(
    title="BossAssistant API",
    description="Department-scoped RAG assistant (HR + Finance).",
    version="0.1.0",
)

app.add_middleware(PolicyAdmission)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    department: Literal["hr", "finance"] | None = None
    mode: RetrievalMode = "fast"
    #: Prior turns, oldest first, as {"role": "user"|"assistant", "content": ...}.
    #: Only used to resolve references in a follow-up; treated as untrusted text.
    history: list[dict[str, str]] | None = None
    include_contexts: bool = False
    """Return the full retrieved chunks. Off by default — the eval harness
    needs them for RAGAS faithfulness; the UI only needs `sources`."""


class Source(BaseModel):
    source: str | None = None
    title: str | None = None
    department: str | None = None
    preview: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    department_routed: Literal["hr", "finance", "both"]
    contexts: list[str] | None = None
    #: True when the assistant declined for lack of supporting policy, as
    #: opposed to answering. Exposed so abstention can be scored directly
    #: instead of inferred from wording that changes with the model.
    abstained: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe: process is up and config loaded."""
    return {"status": "ok", "env": settings.app_env}


@app.get("/ready")
def ready(response: Response) -> dict[str, str]:
    """Readiness requires a readable, published policy gate, not just a live DB."""
    try:
        state = gate.snapshot()
        if state.get("open") is True and not state.get("owner"):
            return {"status": "ready", "neo4j": "ok"}
    except Exception:
        pass
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "not_ready", "neo4j": "closed_or_unavailable"}


class DemoTokenRequest(BaseModel):
    persona: str = "employee"


class DemoTokenResponse(BaseModel):
    token: str
    persona: str
    groups: list[str]
    expires_in: int


@app.get("/auth/personas")
def personas() -> dict[str, object]:
    """List the demo identities, so the UI does not hardcode them."""
    if not settings.enable_demo_auth:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="demo auth is not enabled")
    return {"personas": [{"id": k, "groups": sorted(v)} for k, v in DEMO_PERSONAS.items()]}


@app.post("/auth/demo", response_model=DemoTokenResponse)
def demo_token(request: DemoTokenRequest) -> DemoTokenResponse:
    """Issue a short-lived token for a demo persona.

    There is no credential check — this is an authentication bypass whose only
    purpose is letting a hosted demo show ACL-scoped retrieval. It 404s unless
    `ENABLE_DEMO_AUTH` is explicitly set.
    """
    token, groups, expires_in = issue_demo_token(request.persona)
    return DemoTokenResponse(
        token=token, persona=request.persona, groups=sorted(groups), expires_in=expires_in
    )


def _sse(event: str, data: object) -> str:
    """One Server-Sent Event.

    `json.dumps` on every payload, including plain token strings: a raw token
    containing a newline would otherwise terminate the event early and silently
    truncate the answer.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/ask/stream")
def ask_stream(
    request: AskRequest,
    principal: Principal = Depends(get_principal),
) -> StreamingResponse:
    """Answer as a Server-Sent Event stream.

    Sources are sent before the first token, because retrieval completes before
    generation begins — showing which policies are being read while the answer
    is still being written is most of the perceived-latency win.

    Errors are delivered as an `error` event rather than an HTTP status: the
    response has already begun with 200 by the time generation can fail, so the
    status line is long gone.
    """

    def events() -> Iterator[str]:
        try:
            for event, data in stream_question(
                request.question,
                request.department,
                mode=request.mode,
                groups=principal.groups,
            ):
                yield _sse(event, data)
        except Exception as exc:  # noqa: BLE001 - must reach the client as an event
            retryable = retryable_generation_error(exc)
            yield _sse(
                "error",
                {
                    "detail": "upstream inference provider unavailable; retry"
                    if retryable
                    else "internal error",
                    "retryable": retryable,
                },
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Without this, nginx and most reverse proxies buffer the whole
            # response and deliver it at once, which defeats the point.
            "X-Accel-Buffering": "no",
        },
    )


@app.on_event("startup")
def _verify_auth_configuration() -> None:
    """Refuse to start without a way to verify tokens or reach online generation."""
    if settings.require_auth and settings.jwt_secret is None:
        raise RuntimeError(
            "REQUIRE_AUTH is true but JWT_SECRET is unset: every caller would be "
            "rejected. Set JWT_SECRET, or set REQUIRE_AUTH=false to run "
            "unauthenticated with all-employees access only."
        )
    if not settings.generation_base_url or not settings.generation_api_key:
        raise RuntimeError("GENERATION_BASE_URL and GENERATION_API_KEY are required for online generation")


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    principal: Principal = Depends(get_principal),
) -> AskResponse:
    """Answer a question using the department-scoped, ACL-filtered RAG pipeline.

    The caller's groups are pushed down into retrieval rather than checked
    against the finished answer: by the time an answer exists, restricted text
    has already entered the prompt.
    """
    try:
        result = answer_question(
            request.question,
            request.department,
            mode=request.mode,
            groups=principal.groups,
            history=request.history,
        )
    except Exception as exc:  # noqa: BLE001 - provider raises bare Exception
        if retryable_generation_error(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="upstream inference provider unavailable; retry",
            ) from exc
        raise
    return AskResponse(
        answer=result["answer"],
        sources=[Source(**source) for source in result["sources"]],
        department_routed=result["department_routed"],
        contexts=result["contexts"] if request.include_contexts else None,
        abstained=result.get("abstained", False),
    )
