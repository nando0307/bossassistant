"""Application configuration.

Loads environment variables from `.env` into a typed Settings object.
Import `settings` anywhere in the codebase to access config:

    from app.config import settings
    driver = GraphDatabase.driver(settings.neo4j_uri, ...)

If a required env var is missing, the app fails fast on startup with a
clear validation error rather than crashing later with a cryptic message.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """BossAssistant runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Neo4j ---
    neo4j_uri: str = Field(..., description="Neo4j AuraDB URI, e.g. neo4j+s://xxx.databases.neo4j.io")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: SecretStr = Field(...)
    neo4j_database: str = Field(default="neo4j")

    # --- NVIDIA AI Endpoints ---
    nvidia_api_key: SecretStr = Field(...)

    # --- LangSmith (optional, for tracing) ---
    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "bossassistant"

    # --- CORS ---
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:5173")

    @property
    def cors_origins_list(self) -> list[str]:
        """Split comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using lru_cache means we instantiate Settings exactly once per process.
    Subsequent calls return the same object — fast, and ensures we don't
    reload .env on every import.
    """
    return Settings()


settings = get_settings()