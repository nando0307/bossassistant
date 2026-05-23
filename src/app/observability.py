from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain_core.runnables import RunnableConfig
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from app.config import settings


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    return value.get_secret_value()


@lru_cache(maxsize=1)
def get_langfuse_handler() -> CallbackHandler | None:
    """Return a Langfuse callback handler when tracing is fully configured."""
    public_key = _secret_value(settings.langfuse_public_key)
    secret_key = _secret_value(settings.langfuse_secret_key)

    if not settings.langfuse_tracing or not public_key or not secret_key:
        return None

    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_host
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host

    get_client()
    return CallbackHandler()


def langchain_config(run_name: str, metadata: dict[str, Any] | None = None) -> RunnableConfig:
    """Build LangChain invoke config with optional Langfuse tracing."""
    handler = get_langfuse_handler()
    config: RunnableConfig = {
        "run_name": run_name,
        "metadata": {
            "app": "bossassistant",
            "environment": settings.app_env,
            **(metadata or {}),
        },
    }
    if handler is not None:
        config["callbacks"] = [handler]
    return config
