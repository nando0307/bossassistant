"""FastAPI application entrypoint.

Exposes a minimal HTTP API for BossAssistant. For now, only `/health`
exists — a liveness check used by container orchestrators (Railway,
Docker, Kubernetes) to know whether the process is up and able to serve
requests. As we add agents, retrieval, and ingestion routes, they hang
off this same `app` instance.
"""
from __future__ import annotations

from fastapi import FastAPI

from app.config import settings

app = FastAPI(
    title="BossAssistant API",
    description="Department-scoped RAG assistant (HR + Finance).",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    Returns 200 OK with a tiny JSON payload as long as the process is
    running and config loaded successfully. Does NOT check Neo4j or
    NVIDIA — those belong in a separate `/ready` (readiness) endpoint
    that we'll add when we wire up the real services.
    """
    return {"status": "ok", "env": settings.app_env}
