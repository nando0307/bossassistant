from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("NEO4J_URI", "neo4j://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-not-used-anywhere-real")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_token(groups: list[str], subject: str = "test-user", **overrides: object) -> str:
    """Mint a signed token for tests.

    Imported after the env defaults above so `settings` picks up JWT_SECRET.
    """
    import datetime

    import jwt

    from app.config import settings

    claims: dict[str, object] = {
        "sub": subject,
        "groups": groups,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5),
    }
    claims.update(overrides)
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm)  # type: ignore[union-attr]


def auth_headers(groups: list[str] | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {make_token(groups or ['all-employees'])}"}
