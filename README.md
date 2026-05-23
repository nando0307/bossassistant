# BossAssistant

Department-scoped RAG assistant for HR and Finance policy questions.

BossAssistant started as a Colab prototype and is now a deployed full-stack app:

- FastAPI backend on Railway
- React/Vite frontend on Vercel
- Neo4j AuraDB for hybrid vector + keyword retrieval
- NVIDIA AI Endpoints for LLM generation and embeddings
- Multi-query retrieval with Reciprocal Rank Fusion
- Optional cross-encoder reranking
- Structured routing across HR, Finance, or both departments
- Optional Langfuse tracing for LLM observability

## Live App

- Frontend: https://bossassistant-my7qsq9aj-nando0307s-projects.vercel.app
- API: https://bossassistant-production.up.railway.app

## API

### Health

```bash
curl https://bossassistant-production.up.railway.app/health
```

### Readiness

```bash
curl https://bossassistant-production.up.railway.app/ready
```

### Ask

```bash
curl -s https://bossassistant-production.up.railway.app/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How much PTO do I accrue per year?","department":"hr"}'
```

Request body:

```json
{
  "question": "How much PTO do I accrue per year?",
  "department": "hr"
}
```

`department` may be `"hr"`, `"finance"`, or `null` for automatic routing.

Response shape:

```json
{
  "answer": "You accrue 15 days of PTO per year...",
  "sources": [
    {
      "source": "HR-001",
      "title": "PTO Policy",
      "department": "hr",
      "preview": "Paid Time Off (PTO) Policy..."
    }
  ],
  "department_routed": "hr"
}
```

## Local Development

### Backend

```bash
uv sync
uv run uvicorn app.api.main:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at:

```text
http://127.0.0.1:5173
```

For local frontend-to-backend calls, create `frontend/.env.local`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

## Environment Variables

Backend variables:

```env
NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=
NEO4J_DATABASE=
NVIDIA_API_KEY=
NVIDIA_CHAT_MODEL=meta/llama-3.1-8b-instruct
NVIDIA_MAX_TOKENS=384
ENABLE_MULTI_QUERY=false
LANGSMITH_API_KEY=
LANGSMITH_TRACING=
LANGSMITH_PROJECT=
LANGFUSE_TRACING=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
APP_ENV=
LOG_LEVEL=
CORS_ORIGINS=
ENABLE_RERANKER=false
```

Frontend variables:

```env
VITE_API_URL=
```

Do not commit real secrets. Use `.env.example` files for placeholders only.

## Architecture

```text
React/Vite frontend
        |
        v
FastAPI /ask endpoint
        |
        v
Department router
        |
        +--> HR retriever     --> Neo4j hr_vector + hr_keyword
        |
        +--> Finance retriever --> Neo4j fin_vector + fin_keyword
        |
        v
NVIDIA LLM answer generation
```

Retrieval flow:

1. Retrieve relevant chunks from Neo4j hybrid search.
2. Optionally generate alternate queries and merge ranked results with Reciprocal Rank Fusion.
3. Optionally rerank with `BAAI/bge-reranker-large`.
4. Generate a grounded answer using retrieved policy chunks.

`ENABLE_MULTI_QUERY=false` is the default for deployed latency. Set it to `true` for notebook-faithful multi-query retrieval experiments.

`NVIDIA_CHAT_MODEL` defaults to `meta/llama-3.1-8b-instruct` for production latency. The original notebook model, `qwen/qwen3-next-80b-a3b-instruct`, can be used for higher-capacity experiments.

When Langfuse is enabled, BossAssistant traces key LangChain runs:

- department routing
- multi-query generation
- department answer generation
- both-department synthesis

Enable tracing with:

```env
LANGFUSE_TRACING=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Deployment

Backend:

- Railway builds the root Dockerfile.
- Pushes to `main` trigger automatic redeploys.
- `CORS_ORIGINS` must include the deployed Vercel frontend URL.

Frontend:

- Vercel builds from the `frontend/` root directory.
- Framework preset: Vite
- Build command: `npm run build`
- Output directory: `dist`
- `VITE_API_URL` points to the Railway API URL.

## Development Checks

Backend:

```bash
uv run python -m compileall src
uv run ruff check src
uv run mypy src
uv run pytest
```

Frontend:

```bash
cd frontend
npm run lint
npm run build
```

Note: the current backend test suite is not populated yet, so `pytest` reports zero collected tests.
