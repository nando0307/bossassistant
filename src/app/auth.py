"""JWT verification and the caller identity that drives ACL-filtered retrieval.

The principal's groups are a *retrieval filter*, not a post-hoc check on the
answer. Filtering after generation is not access control: the restricted text
has already been read by the model and can leak through paraphrase, refusal
wording, or a citation list. Groups are threaded down into `retrieve()` so
restricted chunks are never fetched in the first place.
"""
from __future__ import annotations

from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

#: `auto_error=False` so a missing header reaches our handler and can be turned
#: into a 401 with a useful message, rather than FastAPI's bare 403.
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    """Who is asking, and what they are allowed to retrieve."""

    subject: str
    groups: frozenset[str]

    def can_read(self, acl_groups: list[str] | None) -> bool:
        """True when this principal shares at least one group with the chunk.

        A chunk carrying no ACL at all is treated as unreadable rather than
        public. Ingestion stamps every document, so an unstamped chunk means
        something went wrong upstream — failing closed keeps that from becoming
        a silent disclosure.
        """
        if not acl_groups:
            return False
        return bool(self.groups.intersection(acl_groups))


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def decode_token(token: str) -> Principal:
    """Verify a bearer token and extract the principal.

    Signature, expiry, issuer, and audience are all verified. `groups` must be
    present and non-empty: a token that authenticates but grants nothing is a
    misconfiguration, and treating it as "no access" would surface as a
    confusing empty answer instead of an explicit error.
    """
    if settings.jwt_secret is None:
        raise _unauthorized("server has no JWT secret configured")
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
        )
    except jwt.ExpiredSignatureError as exc:
        raise _unauthorized("token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise _unauthorized(f"invalid token: {exc}") from exc

    subject = claims.get("sub")
    groups = claims.get("groups")
    if not subject:
        raise _unauthorized("token is missing the 'sub' claim")
    if not isinstance(groups, list) or not groups or not all(isinstance(g, str) for g in groups):
        raise _unauthorized("token is missing a non-empty 'groups' claim")
    return Principal(subject=subject, groups=frozenset(groups))


def get_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    """FastAPI dependency resolving the caller.

    With `require_auth=False` an unauthenticated caller becomes an anonymous
    principal holding only `all-employees` — enough to exercise the app locally
    without handing out restricted policies. It never grants more than the
    least-privileged real user.
    """
    if credentials is None or not credentials.credentials:
        if settings.require_auth:
            raise _unauthorized("missing bearer token")
        return Principal(subject="anonymous", groups=frozenset({"all-employees"}))
    return decode_token(credentials.credentials)
