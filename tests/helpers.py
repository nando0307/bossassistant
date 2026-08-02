"""Shared test helpers.

Deliberately not in `conftest.py`: pytest gives that file a special import
path, so importing from it directly makes mypy resolve the same source under
two module names ("conftest" and "tests.conftest") and fail the type check
that CI runs over `tests/`.
"""
from __future__ import annotations

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
