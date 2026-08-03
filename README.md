# BossAssistant

[![CI](https://github.com/nando0307/bossassistant/actions/workflows/ci.yml/badge.svg)](https://github.com/nando0307/bossassistant/actions/workflows/ci.yml)

Department-scoped RAG assistant for HR and Finance policy questions.

BossAssistant started as a Colab prototype and is now a deployed full-stack app over a
77-document policy corpus:

- FastAPI backend on Railway, React/Vite frontend on Vercel
- **JWT-scoped retrieval** — the caller's groups filter chunks *during* retrieval
- Neo4j AuraDB hybrid vector + keyword retrieval, plus a **GraphRAG entity graph**
- Incremental ingestion with content hashing, deletes, and policy supersession
- Indirect prompt-injection defence and measured abstention
- SSE streaming, an ACL-partitioned semantic cache, per-request cost accounting
- A 90-case eval suite with a statistical (paired McNemar) regression gate

## What This Demonstrates

- Access-controlled retrieval, with the recall cost of post-ANN filtering measured
  rather than assumed
- Two retrieval strategies behind one interface, compared on the same eval suite
- Ingestion that survives a changing corpus: idempotent re-runs, real deletes,
  superseded policies dropped at query time
- An eval *process* — frozen baselines, a nightly gate that only fails on changes
  clearing the noise floor
- Full-stack deployment with Railway, Vercel, Docker, and GitHub Actions CI

For a recruiter-facing project summary, see [docs/portfolio.md](docs/portfolio.md).

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

### Demo personas (when `ENABLE_DEMO_AUTH=true`)

```bash
curl -s $API/auth/personas
curl -s -X POST $API/auth/demo -H 'Content-Type: application/json' \
  -d '{"persona":"executive"}'
```

Retrieval is ACL-filtered, so the same question answers differently per persona:

| persona | "What approvals are required for an acquisition?" |
|---|---|
| `employee` | *"Not covered in policy. I checked FIN-008, FIN-013, FIN-014, FIN-033."* |
| `executive` | *"Board approval is required for any acquisition or divestiture…"* (cites FIN-037) |

### Ask

```bash
curl -s https://bossassistant-production.up.railway.app/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How much PTO do I accrue per year?","department":"hr"}'
```

All `/ask` requests require a bearer token — the caller's `groups` claim filters
retrieval, so an unauthenticated request is a 401 rather than an unscoped answer.

Request body:

```json
{
  "question": "How much PTO do I accrue per year?",
  "department": "hr",
  "mode": "fast",
  "history": [{"role": "user", "content": "..."}]
}
```

`mode` is `"fast"`, `"deep"`, or `"graph"` (entity-graph retrieval). `history` is
optional and used only to resolve references in a follow-up such as "what about
internationally?".

### Ask, streamed

```bash
curl -N -X POST $API/ask/stream -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" -d '{"question":"How much PTO do I accrue?"}'
```

Server-Sent Events: `meta`, then `sources_formatted`, then `token` events, then `done`.
Sources arrive before the first token. Failures arrive as an `error` event with a
`retryable` flag, since the 200 is already sent by the time generation can fail.

`department` may be `"hr"`, `"finance"`, or `null` for automatic routing.

`mode` may be `"fast"` or `"deep"`. Fast mode is the production default. Deep mode enables multi-query retrieval and uses the reranker when `ENABLE_RERANKER=true`.

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
# qwen/qwen3-next-80b-a3b-instruct reached NVIDIA end-of-life and now returns 410.
NVIDIA_CHAT_MODEL=nvidia/nemotron-3-super-120b-a12b
NVIDIA_DEEP_CHAT_MODEL=nvidia/nemotron-3-ultra-550b-a55b
NVIDIA_MAX_TOKENS=384

# --- Auth. Retrieval is ACL-filtered, so these are not optional. ---
# The app refuses to start with REQUIRE_AUTH=true and no JWT_SECRET rather than
# serving restricted policy to anonymous callers.
JWT_SECRET=
JWT_ALGORITHM=HS256
JWT_ISSUER=bossassistant
JWT_AUDIENCE=bossassistant-api
REQUIRE_AUTH=true
# Mints tokens for demo personas with NO credential check. Off by default; only
# enable on a deployment holding synthetic policy.
ENABLE_DEMO_AUTH=false

ENABLE_SEMANTIC_CACHE=true
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
        +--> Multi-question splitter / clarification guard
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

Routing behavior:

- Clear HR questions route to HR.
- Clear Finance questions route to Finance.
- Cross-department questions route to both departments.
- Bundled prompts are split and answered one question at a time.
- Vague standalone questions, such as "How much do I get?", ask for clarification instead of guessing.

Fast mode is the default for deployed latency. Use request-level `"mode": "deep"` for notebook-faithful multi-query retrieval experiments.

`NVIDIA_CHAT_MODEL` controls fast/default answer generation, routing, and multi-query
generation. `NVIDIA_DEEP_CHAT_MODEL` controls deep-mode answer generation.

Two measured constraints on model choice:

- The configured chat model is a **reasoning** model. Its thinking trace breaks
  LangChain's `with_structured_output`, which is why graph extraction, the GraphRAG map
  step, and RAGAS scoring each pin a non-reasoning model instead.
- It also emits its whole answer after thinking rather than incrementally, so
  time-to-first-token is ~3.2s versus ~0.4s for `mistralai/mistral-nemotron`. That is
  why SSE streaming buys little here — see `docs/portfolio.md`.

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
- **`JWT_SECRET` must be set.** The app refuses to start with `REQUIRE_AUTH=true` and no
  secret, rather than serving restricted policy to anonymous callers.
- Set `ENABLE_DEMO_AUTH=true` only on a demo deployment. It mints tokens for named
  personas with no credential check, which is what makes the hosted demo able to show
  ACL-scoped retrieval — and an authentication bypass anywhere else.

Frontend:

- Vercel builds from the `frontend/` root directory.
- Framework preset: Vite
- Build command: `npm run build`
- Output directory: `dist`
- `VITE_API_URL` points to the Railway API URL.

## Roadmap

### Retrieval Modes

Production defaults to a low-latency path so the deployed app stays usable:

```env
ENABLE_RERANKER=false
NVIDIA_CHAT_MODEL=nvidia/nemotron-3-super-120b-a12b
NVIDIA_DEEP_CHAT_MODEL=nvidia/nemotron-3-ultra-550b-a55b
```

Current mode behavior:

- Use `"fast"` for the default deployed UI.
- Use `"deep"` for RAGAS evaluation and portfolio writeups.
- In `"deep"` mode, multi-query retrieval is enabled for that request.
- In `"deep"` mode, BossAssistant uses `NVIDIA_DEEP_CHAT_MODEL` for final answer generation.
- Reranking runs in `"deep"` mode when `ENABLE_RERANKER=true`.
- Replace the local `BAAI/bge-reranker-large` cross-encoder with a hosted reranker before making deep mode production-default.

Reranking is an **optional extra**, not a base dependency:

```bash
uv sync --extra rerank    # only needed to run with ENABLE_RERANKER=true
```

`sentence-transformers` pulls torch and transformers (~470MB installed on macOS, and
15 additional CUDA wheels on Linux) plus a ~1.5GB model download on first use. It runs
only in deep mode with `ENABLE_RERANKER=true`, which is off by default, so the serving
image does not carry it. `get_reranker()` raises with install instructions if the flag
is set without the extra.

## Evaluation

90 cases: 75 factoid in `evals/questions.jsonl` covering all 77 documents, plus 15
multi-hop questions in `evals/questions_multihop.jsonl` whose answers span 4-19
documents each.

Because retrieval is ACL-filtered, the harness needs a token. Baselines use one holding
every group, so the numbers measure retrieval quality rather than ACL selectivity —
`scripts/measure_acl_recall.py` covers that separately.

```bash
TOKEN=$(uv run python scripts/mint_token.py \
  --groups all-employees managers hr-team finance-team executives)

uv run python scripts/run_eval.py --api-url http://127.0.0.1:8000 --token "$TOKEN"
```

Compare a run against the frozen baseline:

```bash
uv run python scripts/eval_gate.py check evals/results_fast.jsonl \
  --baseline evals/baseline_fast.json
```

The gate uses an exact one-sided **McNemar** test on the cases that changed verdict,
because the two runs are paired. It reports a Newcombe interval for effect size and
fails only when a change clears the measured noise floor — at n=75 a 5-point threshold
fires on 4 cases flipping, which is inside the variance provider 503s alone produce.
Latency and RAGAS are reported but never gated.

Run against a local API:

```bash
uv run python scripts/run_eval.py --api-url http://127.0.0.1:8000 --token "$TOKEN"
```

Run against the deployed Railway API:

```bash
uv run python scripts/run_eval.py --api-url https://bossassistant-production.up.railway.app --mode fast
```

Run a small deep-mode smoke eval with a bounded request timeout:

```bash
uv run python scripts/run_eval.py --api-url https://bossassistant-production.up.railway.app --mode deep --limit 2 --timeout 45
```

Score answer quality with RAGAS (faithfulness + answer relevance):

```bash
uv run python scripts/run_eval.py --api-url http://127.0.0.1:8000 --mode deep --ragas
```

`--ragas` makes the harness request `include_contexts` so each answer is graded
against the untruncated chunks the model actually retrieved. Scoring is billed
LLM work — it runs a judge model per case, so keep it off routine smoke runs.
Override the judge with `--ragas-model` / `--ragas-embed-model`; both defaults
must be models your NVIDIA account can reach, or the run fails loudly rather
than reporting empty scores.

The script records:

- request success
- latency, including p50 and p95
- expected vs. actual routed department
- expected source coverage
- required answer terms via `must_include`
- forbidden answer terms via `must_not_include`
- model answer text for manual review
- per-case RAGAS scores under `ragas` when `--ragas` is set

Generated eval output is written to `evals/results.jsonl` and is intentionally git-ignored.

## Scripts

| script | purpose |
|---|---|
| `scripts/ingest.py` | Incremental corpus ingest. Content-hashed, deletes orphans, `--dry-run` shows the diff without writing or embedding. |
| `scripts/graph_index.py` | Build the GraphRAG entity graph: extract, resolve, cluster, summarise, embed. Staged and resumable. |
| `scripts/mint_token.py` | Mint a dev JWT for a set of groups. |
| `scripts/run_eval.py` | Run the eval suite against a live API, optionally RAGAS-scored. |
| `scripts/eval_gate.py` | Freeze a baseline, or check a run against one with a paired McNemar test. |
| `scripts/measure_acl_recall.py` | Price the recall cost of post-ANN ACL filtering against exact cosine. |
| `scripts/injection_drill.py` | Plant a hostile policy document, verify the assistant ignores it, remove it. |

Re-running `ingest.py` with no corpus change writes nothing:

```bash
uv run python scripts/ingest.py --dry-run
# HR: 39 documents -> 73 chunks (new 0, changed 0, unchanged 73, deleted 0)
```

## Development Checks

Backend:

```bash
uv run python -m compileall src scripts tests
uv run ruff check src scripts tests
uv run mypy src scripts tests
uv run pytest
```

Frontend:

```bash
cd frontend
npm run lint
npm run build
```

GitHub Actions runs these checks on every push to `main`.
