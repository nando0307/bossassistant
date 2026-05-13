# syntax=docker/dockerfile:1.7

# =========================================================
# Stage 1: builder — install deps with uv into a venv
# =========================================================
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Copy ONLY the dep manifest first. This layer is cached
# unless pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./

# Install deps only (not the project itself).
RUN uv sync --frozen --no-install-project --no-dev

# Copy project metadata files referenced by pyproject.toml.
# Kept in their own layer so README edits don't bust the deps cache.
COPY README.md LICENSE ./

# Copy source.
COPY src ./src

# Install the project itself into the venv.
RUN uv sync --frozen --no-dev

# =========================================================
# Stage 2: runtime — minimal image, just Python + venv + src
# =========================================================
FROM python:3.12-slim AS runtime

RUN groupadd --system --gid 1001 app \
 && useradd --system --uid 1001 --gid app --no-create-home app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
