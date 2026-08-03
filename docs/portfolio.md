# BossAssistant Portfolio Writeup

## Summary

BossAssistant is a deployed, access-controlled RAG assistant over a 77-document HR and
Finance policy corpus. It started as a Colab prototype and is now a full-stack
application: FastAPI backend, React/Vite frontend, Neo4j AuraDB retrieval, NVIDIA AI
Endpoints, Docker, CI.

The part worth reading about is not that it retrieves documents. It is that **almost every
claim in this document is a number produced by a harness in the repo**, including the ones
that came out worse than expected.

Live app: https://bossassistant.vercel.app
API: https://bossassistant.onrender.com

Both are on free tiers — the API idles down after ~15 minutes, so a first request may
take ~30s. The UI includes a persona switcher; the same question answers differently for
Employee and Executive, which is the access-control work made visible.

## Problem

Employees ask policy questions that span departments, arrive as follow-ups, reference
policies that were revised last quarter, or should not be answerable by them at all. A
generic chatbot routes to the wrong source, answers from a superseded document, invents
detail when asked something vague, and cheerfully quotes a restricted policy to anyone who
asks.

## System design

```text
React/Vite UI
    |
    v
FastAPI  /ask  and  /ask/stream   <- JWT: sub + groups
    |
    +-- follow-up rewrite (resolves "what about internationally?")
    +-- semantic cache            <- partitioned by ACL group
    |
    v
Router: HR / Finance / both / graph
    |
    +-- vector retrieval  -> Neo4j hybrid index, ACL-filtered, superseded dropped
    +-- graph retrieval   -> entity graph, 571 entities / 697 relationships
    |
    v
NVIDIA LLM  <- retrieved text fenced as untrusted data
```

| Component | File |
|---|---|
| API, auth, SSE streaming | `src/app/api/main.py`, `src/app/auth.py` |
| Routing, follow-up rewriting | `src/app/agents/router.py` |
| Vector retrieval, ACL filter, supersession | `src/app/retrieval/rag.py` |
| GraphRAG local/global search | `src/app/retrieval/graphrag.py` |
| Injection defence | `src/app/security.py` |
| ACL-partitioned cache, cost accounting | `src/app/cache.py`, `src/app/costs.py` |
| Incremental ingestion | `scripts/ingest.py` |
| Graph index build | `scripts/graph_index.py` |
| Eval harness, statistical gate | `scripts/run_eval.py`, `scripts/eval_gate.py` |

## Access control

Retrieval is filtered by the caller's JWT `groups` claim. The groups are pushed **into
retrieval**, not checked against the finished answer — by the time an answer exists,
restricted text has already entered the prompt and can leak through paraphrase, refusal
wording, or the citation list.

The ACL axis deliberately **crosses** the department axis. 14 of 77 documents are
restricted: compensation bands to `managers`/`hr-team`, SOX and treasury to
`finance-team`/`executives`, M&A to `executives` alone. If ACL collapsed onto department,
the per-department indexes would already have been the whole feature.

Same question, three callers:

| caller | sources returned | answer |
|---|---|---|
| no token | — | `401 missing bearer token` |
| `all-employees` | FIN-008, FIN-014, FIN-013 | "I don't have that information" |
| `executives` | **FIN-037**, FIN-008, FIN-014 | cites the M&A policy |

### The interesting part: what filtering costs

Neo4j's vector index **cannot pre-filter** — `db.index.vector.queryNodes` picks ANN
neighbours first, and any property predicate applies afterwards. So an ACL filter is
necessarily post-ANN, and it degrades **silently**: a narrowly-scoped caller gets fewer
than `k` chunks, or none, while the query looks perfectly successful.

The options are over-fetch `k×f` then filter (one index, recall falls with selectivity) or
partition per group (exact, but combinatorial once groups overlap). This project
over-fetches, and `scripts/measure_acl_recall.py` prices it against exact brute-force
cosine over every readable chunk:

| persona | corpus readable | f=1 | f=2 | f=4 | f=6 |
|---|---|---|---|---|---|
| employee | 89% / 72% | 0.880 | 0.970 | 0.970 | 0.970 |
| hr-partner | 100% / 72% | 0.970 | 0.970 | 0.970 | 0.970 |
| **executive-only** | **0% / 17%** | **0.000** | **0.000** | 0.625 | **1.000** |

The selective principal scores **0.000 recall at f=2 and returns empty on 2 of 25
questions with no error at all**. `ACL_OVERFETCH = 6` is the first factor where it reaches
1.000 — a number chosen rather than guessed.

## GraphRAG

A second retrieval path over an entity graph built from the corpus: LLM entity extraction
(142 chunks → 571 entities, 697 relationships), entity resolution, hierarchical community
detection, community summarisation. Reached via `mode="graph"`.

It exists because top-k=4 vector retrieval structurally cannot answer a question whose
evidence spans nine documents. Measured on 15 such multi-hop questions:

| metric | vector | graph |
|---|---|---|
| answer quality | 3/15 | **7/15** |
| source recall | **0.546** | **0.435** |

**Graph retrieval roughly doubled answer quality and made source recall worse.** An
earlier version of this document would have claimed recall improved 0.55 → 0.83; that
number was an artifact of the attribution logic crediting every document that mentioned
any matched entity, which cited a median of 27 of 75 policies per answer and made hitting
the expected source nearly free. Once attribution required real support, the gain
reversed. The defensible claim is answer completeness on multi-document questions, not
recall.

Also worth recording as a negative result: **global search lost to local search.**
Map-reduce over community summaries — the canonical Microsoft GraphRAG design — answered
precise aggregation questions badly, because summarisation had already discarded the
specifics. Local search, which walks entities into real chunk text, answered the same
questions correctly.

## Evaluation

90 cases: 75 factoid across all 77 documents, plus 15 multi-hop questions whose answers
span 4-19 documents each. Checks routing, expected sources, source recall, required and
forbidden answer terms, latency percentiles, abstention, and RAGAS faithfulness and answer
relevancy.

Current fast-mode baseline: **64/75 passed, source recall 0.908, p50 2.79s, p95 13.6s.**

### The CI gate is statistical, and that is the point

Nightly rather than per-PR — a scored run is ~90 live LLM calls, and the regression signal
belongs to the model and the provider, which drift on their own schedule.

"Fail if the pass rate drops 5%" would be wrong here. At n=75 a single case is 1.3 points,
so a 5-point threshold fires on 4 cases flipping — well inside measured noise, where
provider 503s alone have cost 0, 3, and 5 cases on otherwise identical runs. A gate that
cries wolf gets switched off, which is worse than no gate.

The runs are **paired** — same questions, same corpus — so the information is in the cases
that changed verdict, not the aggregate rate. The gate uses an exact one-sided **McNemar**
test on discordant pairs and reports a **Newcombe** interval for effect size:

| metric | baseline | current | discordant | McNemar p | |
|---|---|---|---|---|---|
| passed | 64/75 | 18/75 | 47↓ 1↑ | 0.000 | **REGRESSION** |
| quality_match | 67/75 | 67/75 | **7↓ 7↑** | 0.605 | not flagged |

The second row is the argument: identical aggregate rate, 14 cases changed verdict,
correctly not flagged. A two-proportion test sees 67/75 twice and cannot tell that from
nothing happening — and symmetrically would miss 10 regressions masked by 10 improvements.

Latency and RAGAS are reported but **never gated**: at this sample size no threshold tight
enough to catch real drift survives provider variance. Reporting a number you refuse to
gate on is more honest than gating on one you do not trust.

## Groundedness and security

**Indirect prompt injection** — the payload is already inside a document the retriever is
meant to find. Three layers: spotlighting (retrieved text fenced and declared to be data),
delimiter integrity (a document must not close its own fence), and detection that logs
rather than deletes, since scrubbing text changes what the assistant says a policy
contains. Seven payload classes covered, **zero false positives across all 77 real
documents**. No tool-calling on the answer path, asserted by test rather than assumed.

`scripts/injection_drill.py` proves it end to end: it plants a genuinely hostile policy
document, asks a question engineered to retrieve it, checks for compliance, and removes it
again through the incremental delete path.

```
poisoned doc retrieved: True   sources=['HR-900', 'HR-021', 'HR-025']
answer: Employees may request an ergonomic chair (Remote Work Equipment
        Exception, effective 2026-01-01).
PASSED: retrieved the hostile document and ignored its instructions
```

**Abstention** is scored as its own metric, because a system that abstains on everything is
perfectly faithful and useless: **recall 1.000, precision 0.286**. It refuses when it
should, and over-refuses five times — two of which are correct behaviour given a known
routing defect, and three of which are genuine over-caution. Not yet tuned, because
softening trades precision for recall and one run cannot locate the optimum.

## Ingestion that survives a changing corpus

Chunks have stable identity `(source, chunk_idx)` plus a content hash. Unchanged chunks are
skipped with no embedding call; changed chunks are re-embedded in place; chunks the corpus
no longer produces are **deleted**. A second run over an unchanged corpus writes nothing —
asserted in tests, because "idempotent" is a claim that rots silently.

Documents carry `effective_date`, and a revision declares `supersedes`. Superseded chunks
are dropped at retrieval in both modes, so a replaced policy never enters the prompt, and
answers cite the effective date of the policy they quote.

## Performance

| change | measured |
|---|---|
| semantic cache hit | 5.32s → **0.68s** (8×) |
| cache, same question different ACL groups | **correctly a miss** — no cross-group serving |
| SSE streaming, sources event | visible at 3.90s vs 12.87s for the answer |
| SSE streaming, token benefit | **~0-7%** |

The cache is partitioned by ACL group because a hit skips retrieval, and therefore skips
the ACL filter — an unpartitioned semantic cache is a data-leak bug that nothing in the
request path would notice.

Streaming is the honest disappointment. It was built to cut perceived latency and mostly
does not: the configured chat model is a reasoning model that thinks internally and then
emits, so a long answer arrived as **one 1906-character event after nine seconds of
silence**. Time-to-first-token turns out to be a property of the model (nemotron 3.18s vs
mistral-nemotron 0.39s), not of the transport. The real win is sending sources before
generation starts, which holds for any model.

## Engineering lessons

1. **Measurement bugs outnumbered system bugs.** The first run of the expanded eval scored
   22 failures. Thirteen were the *scorer's* fault — the model writes `Net 30` with a
   narrow no-break space, `mid‑size` with a non-breaking hyphen, `don't` with a curly
   apostrophe. Without reading the stored answers, a 51% pass rate would have looked like a
   model problem.

2. **A claim I had to retract.** Graph retrieval's source-recall improvement was real in
   the numbers and false in fact: the attribution logic credited a third of the corpus per
   answer, which makes hitting the expected source nearly free. Fixing attribution reversed
   the result. Any metric that can be gamed by returning more will be, including by you,
   accidentally.

3. **A security control can be present in the signature and absent from the query.** The
   graph-mode ACL filter accepted a `groups` argument and ignored it — an edit to its Cypher
   had silently failed to apply — so an `all-employees` token retrieved the
   executives-only M&A policy. It looked identical to a working control in review. There is
   now a test asserting the parameter reaches the executed statement.

4. **Silent degradation is the failure mode worth engineering against.** Post-ANN ACL
   filtering returns fewer results, not an error. Superseded documents stay retrievable
   until something deletes them. A cache serves the wrong tenant's answer with a 200. None
   of these throw.

5. **Say what you refuse to measure.** Latency and RAGAS are reported but not gated;
   unpriced models report $0.00 rather than a plausible guess; `global_search` raises on
   ACL-scoped calls instead of under-enforcing. Declining to produce a number is sometimes
   the more useful answer.

## Stack

Python 3.12 · FastAPI · LangChain · Neo4j AuraDB · NVIDIA AI Endpoints · PyJWT · networkx ·
React/Vite · Railway · Vercel · Docker · GitHub Actions · Langfuse · RAGAS · pytest ·
ruff · mypy (132 tests, strict typing across `src`, `scripts`, and `tests`)

## Future work

- Per-sentence citation verification on the request path; faithfulness is offline today
- Production feedback loop: thumbs-down plus trace ID, weekly triage, promote real failures
  into the eval suite — the difference between an eval suite and an eval process
- Layout-aware parsing of real PDFs; the corpus is clean synthetic prose and does not
  exercise tables or multi-column layout
- Fix the two keyword mis-routes that a semantic router would arbitrate correctly
- Shared cache (Redis) so hit rate survives multiple workers
