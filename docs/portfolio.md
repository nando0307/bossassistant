# BossAssistant Portfolio Writeup

## Summary

BossAssistant is a deployed department-scoped RAG assistant for HR and Finance policy questions. It started as a Google Colab prototype and was converted into a full-stack application with a FastAPI backend, React/Vite frontend, Neo4j AuraDB retrieval layer, NVIDIA AI Endpoints, Docker deployment, observability hooks, and CI.

Live app: https://bossassistant-my7qsq9aj-nando0307s-projects.vercel.app

API: https://bossassistant-production.up.railway.app

## Problem

Employees often ask policy questions that span multiple internal departments. A generic chatbot can answer too broadly, route to the wrong policy source, or invent details when the user asks something vague like "How much do I get?"

BossAssistant solves a narrower version of that problem: answer HR and Finance policy questions using only retrieved policy context, route each question to the right department, and return cited sources.

## System Design

```text
React/Vite UI
    |
    v
FastAPI /ask endpoint
    |
    v
Question router
    |
    +-- HR retriever      -> Neo4j hr_vector + hr_keyword
    |
    +-- Finance retriever -> Neo4j fin_vector + fin_keyword
    |
    v
NVIDIA LLM answer generation
```

Core components:

- `src/app/api/main.py`: FastAPI app, health checks, CORS, `/ask` endpoint
- `src/app/agents/router.py`: HR/Finance/both routing, multi-question splitting, vague-question clarification
- `src/app/retrieval/rag.py`: Neo4j hybrid retrieval, optional multi-query/RRF, optional reranking, answer generation
- `scripts/run_eval.py`: lightweight deployed API evaluation harness
- `.github/workflows/ci.yml`: backend and frontend CI

## Retrieval And Routing

BossAssistant uses separate Neo4j indexes for HR and Finance documents:

- HR: `hr_vector`, `hr_keyword`
- Finance: `fin_vector`, `fin_keyword`

The API supports three routing modes:

- Explicit HR or Finance if the user chooses a department
- Automatic single-department routing for clear questions
- Both-department routing for cross-functional questions

It also splits bundled prompts into smaller questions. For example:

```text
How much PTO do I accrue per year? What is the hotel budget for business travel to NYC?
```

is answered as two separate subquestions instead of being treated as one ambiguous query.

## Production Latency Tradeoff

The original notebook used multi-query generation, Reciprocal Rank Fusion, and local BGE cross-encoder reranking. That produces a deeper RAG pipeline, but it is too slow for a deployed demo on a small cloud service.

Production defaults to a fast configuration:

```env
ENABLE_MULTI_QUERY=false
ENABLE_RERANKER=false
NVIDIA_CHAT_MODEL=meta/llama-3.1-8b-instruct
NVIDIA_MAX_TOKENS=384
```

The planned next version is to expose request modes:

- `fast`: default deployed app path
- `deep`: multi-query plus reranking for evaluation and portfolio experiments

## Evaluation

The project includes 8 evaluation cases covering:

- HR-only questions
- Finance-only questions
- Cross-department questions
- Bundled multi-question prompts
- Vague questions that should ask for clarification

The eval harness checks:

- request success
- latency
- routed department
- expected source IDs
- required answer terms
- forbidden answer terms

Latest deployed run:

```text
cases=8 ok=8
department_matches=8/8
source_hits=8/8
quality_matches=8/8
fully_passed=8/8
avg_latency_seconds=1.81
```

## Engineering Lessons

1. Notebook pipelines need product boundaries.

   The Colab version was useful for proving the RAG idea, but deployment required typed config, package structure, API models, Docker, CI, and environment-specific behavior.

2. Routing matters as much as retrieval.

   Many failures were not retrieval failures. They were question-shape failures: cross-department prompts, bundled questions, or vague references. Splitting and clarification improved behavior more than simply adding more retrieval.

3. Eval checks need to catch wrong-but-plausible answers.

   A source hit is not enough. One PTO answer retrieved the right document but briefly miscalculated the annual accrual. The eval harness now includes forbidden answer terms so this kind of regression fails.

4. Production demos need latency discipline.

   Multi-query and reranking are valuable, but they made the deployed app feel slow. The production path now prioritizes responsiveness, while deeper retrieval remains available as an experimental direction.

## Stack

- Python 3.12
- FastAPI
- LangChain
- Neo4j AuraDB
- NVIDIA AI Endpoints
- React/Vite
- Railway
- Vercel
- Docker
- GitHub Actions
- Langfuse optional tracing

## Future Work

- Add explicit `fast` and `deep` request modes
- Add RAGAS or LLM-as-judge scoring for answer faithfulness
- Replace local reranking with a hosted reranker for production deep mode
- Add document ingestion CLI for new policy files
- Add persistent frontend conversation history
