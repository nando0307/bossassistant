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

from typing import Literal

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.router import answer_question
from app.config import settings
from app.db import verify_connectivity
from app.retrieval.rag import RetrievalMode  # single definition; a local copy silently drifted

app = FastAPI(
    title="BossAssistant API",
    description="Department-scoped RAG assistant (HR + Finance).",
    version="0.1.0",
)

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


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe: process is up and config loaded."""
    return {"status": "ok", "env": settings.app_env}


@app.get("/ready")
def ready(response: Response) -> dict[str, str]:
    """Readiness probe: Neo4j is reachable.

    Returns 200 with {"status": "ready", "neo4j": "ok"} on success.
    Returns 503 with {"status": "not_ready", "neo4j": "down"} on failure.
    503 (Service Unavailable) is the canonical status for "I'm alive
    but can't serve traffic right now" — load balancers will stop
    routing to this instance until it returns 200 again.
    """
    if verify_connectivity():
        return {"status": "ready", "neo4j": "ok"}
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "not_ready", "neo4j": "down"}


#: Upstream inference failures that a caller should retry rather than treat as a
#: bug in this service. NVIDIA returns "ResourceExhausted: Worker local total
#: request limit reached (107/32)" and bare 502s under sustained load; surfacing
#: those as 500 told every client the request was malformed and defeated retry
#: logic, costing whole eval runs.
_UPSTREAM_RETRYABLE = ("[502]", "[503]", "[504]", "ResourceExhausted", "Too Many Requests")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question using the department-scoped RAG pipeline."""
    try:
        result = answer_question(request.question, request.department, mode=request.mode)
    except Exception as exc:  # noqa: BLE001 - provider raises bare Exception
        message = str(exc)
        if any(marker in message for marker in _UPSTREAM_RETRYABLE):
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
    )
