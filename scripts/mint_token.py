"""Mint a development JWT for the ACL-filtered /ask endpoint.

    uv run python scripts/mint_token.py --groups all-employees
    uv run python scripts/mint_token.py --groups executives finance-team --ttl 60

Development only. In a real deployment these come from the identity provider,
and `groups` mirrors directory group membership rather than being hand-typed.
"""
from __future__ import annotations

import argparse
import datetime

import jwt
from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint a dev JWT.")
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--subject", default="dev-user")
    parser.add_argument("--ttl", type=int, default=60, help="Minutes until expiry.")
    args = parser.parse_args()

    load_dotenv()
    from app.config import settings
    from app.corpus import ALL_GROUPS

    if settings.jwt_secret is None:
        raise SystemExit("JWT_SECRET is not set; add it to .env before minting tokens")

    unknown = sorted(set(args.groups) - ALL_GROUPS)
    if unknown:
        # Not fatal — a real directory will have groups this corpus never uses —
        # but a typo'd group silently grants nothing, which is confusing to debug.
        print(f"warning: {unknown} match no document in the corpus; known groups: {sorted(ALL_GROUPS)}")

    token = jwt.encode(
        {
            "sub": args.subject,
            "groups": args.groups,
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=args.ttl),
        },
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    print(token)


if __name__ == "__main__":
    main()
