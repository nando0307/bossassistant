"""Neo4j driver lifecycle.

Provides a single shared `Driver` instance for the application. Neo4j's
official Python driver is thread-safe and connection-pooled internally,
so creating one driver per process (not per request) is the right move.

Usage:
    from app.db import get_driver
    driver = get_driver()
    with driver.session() as session:
        result = session.run("RETURN 1 AS n")
"""
from __future__ import annotations

from functools import lru_cache

from neo4j import Driver, GraphDatabase

from app.config import settings


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    """Return a process-wide Neo4j driver.

    Cached via lru_cache so we instantiate the driver exactly once.
    The driver maintains its own connection pool internally — do NOT
    create one per request, that defeats the pool entirely.
    """
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
