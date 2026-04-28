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

from fastapi import FastAPI, Response, status

from app.config import settings
from app.db import verify_connectivity

app = FastAPI(
    title="BossAssistant API",
    description="Department-scoped RAG assistant (HR + Finance).",
    version="0.1.0",
)


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
