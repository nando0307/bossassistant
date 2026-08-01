# BossAssistant — Evaluation Plan

Goal: replace unverifiable resume claims with measured numbers.

Last updated: 2026-07-31

## Where things stand

### Working and verified

- **Corpus is reproducible.** `scripts/ingest.py` was an empty scaffold; the 16 policy
  documents and ingestion logic existed only in `notebooks/01_prototype.ipynb`. Ported
  into the script, which now pulls `INDEX_CONFIG` and `get_embedder()` from
  `app.retrieval.rag` so index names cannot drift from query time. `--reset` guards the
  `DETACH DELETE`; re-running without it refuses rather than duplicating chunks.
  Ingested: 8 HR docs → 12 chunks, 8 Finance docs → 13 chunks.
- **Chat model** set to `nvidia/nemotron-3-super-120b-a12b` in `.env`. The previous
  default (`qwen/qwen3-next-80b-a3b-instruct`) hit NVIDIA end-of-life 2026-07-27 and
  returns 410. **Railway still needs this env var set.**
- **`/ask` gained `include_contexts`** (default off). Returns untruncated retrieved
  chunks for scoring; the 240-char `preview` in `sources` would score grounded answers
  as unfaithful.
- **`evals/questions.jsonl` grown from 8 to 75 cases**, covering all 16 corpus
  documents, plus cross-department, out-of-corpus refusal, adversarial false-premise,
  and vague-question cases. Two cases are deliberate routing traps (see Known broken 1).
- **Answer relevancy fixed and verified discriminating** (was unusable, two passing cases
  scoring 0.0). Root cause was not the embedding model: ragas' `ResponseRelevancy`
  compares the original question against questions it generates back from the answer,
  but embeds the first with `embed_query` and the rest with `embed_documents`. On an
  asymmetric query/passage model like `nv-embedqa-e5-v5` that scores two paraphrases of
  the same question as unrelated. `run_eval.py` now routes both sides through the query
  encoder. Verified on a good / hallucinated / off-topic triple: **0.809 / 0.636 /
  0.266**. The good-vs-hallucinated gap is small by design — relevancy measures whether
  the answer addresses the question, and a wrong PTO answer is still about PTO. Catching
  that is faithfulness' job.
- **Two router defects found and fixed** by the expanded eval (see Fixed below).

### Measured baseline — fast mode, n=75

The corpus has since grown from 16 to 75 documents (see GraphRAG below), which made
retrieval harder and moved this baseline down. Both numbers are real; they measure
different corpora.

| Metric | 16-doc corpus | 75-doc corpus |
|---|---|---|
| Functional pass rate | 71/75 (94.7%) | **64/75 (85.3%)** |
| Routing correct | 71/75 | 71/75 |
| Source hits | 71/75 | 67/75 |
| Source recall | not measured | 0.907 |
| p50 latency | 2.33–2.47s | 2.81s |
| p95 latency | 8.17–15.01s | 15.74s |

**Faithfulness 0.762 (n=70/73), answer relevancy 0.677 (n=73/73)** — see RAGAS below.

The 9-case drop at 75 documents is genuine retrieval competition, not a regression:
`fin_external_auditor` loses FIN-007 (KPMG) to three new audit-adjacent documents, and
`hr_medical_plan_tiers` and `fin_receipt_threshold` get pushed out of the top-4. Three
of the failures are attribution ambiguity rather than wrong answers — the new FIN-031
restates the $125/$75 per diems, so the retriever cites a genuinely correct alternate
source while `expected_sources` names only FIN-001.

Baseline moved as defects were fixed, measured on the same 75 cases:

| Run | Routing | Source hits | Fully passed |
|---|---|---|---|
| 1 — as found | 52/75 | 53/75 | 38/75 |
| 2 — after vagueness fix | 71/75 | 71/75 | 53/75 |
| 3 — after scorer typography fix | 72/75 | 72/75 | 70/75 |
| 4 — after `%` fix | 67/75\* | 67/75\* | 66/75\* |

\* Run 4 lost 5 cases to upstream `503 ResourceExhausted`, not to regressions. The
deterministic figure with a healthy provider is **71/75**: runs 3 and 4 fail the same
4 cases, and run 4 additionally fixed `hr_hdhp_premium_share`.

p95 is no longer dominated by `multi_question_bundle` — at n=75 it is one case in
seventy-five rather than one in eight. The old 48.4s p95 was a small-sample artifact.

### Fixed during this pass

1. **Keyword-absence was treated as vagueness.** `is_vague_subquestion()` returned True
   for any question missing one of 26 hardcoded routing terms, so `answer_question()`
   refused it with "Please clarify" before retrieval ever ran. This rejected **19 of 75
   answerable questions (25%)** — "When does the fiscal year end?", "What medical plan
   tiers does the company offer?", "Who are our external auditors?". Vagueness now means
   a dangling demonstrative (`this`/`it`/`these`/`those`) with no topic, or an explicit
   `VAGUE_PATTERNS` entry. The LLM router already existed to handle unfamiliar wording;
   the guard was short-circuiting it.
2. **Substring routing matched inside words.** `_contains_any()` used raw `in`, so
   "three" and "through" contain "hr" → HR, and "approval"/"happens" contain the "ap"
   finance term → Finance. Now left-anchored to a word boundary (keeping plurals:
   "expenses", "invoices"), and "ap" is spelled "accounts payable". The repo's own test
   had flagged this as a known tradeoff to revisit.

Both were found only because the eval set grew past the 8 cases that happened to
contain the right keywords.

### Known broken

1. **Keyword pre-routing mis-routes 2/75 (2.7%).** `route_question_fast()` fires on a
   single keyword before the LLM router is consulted, so an HR question phrased with a
   Finance word goes to the wrong index and then correctly answers "I don't have that
   information": "What is my annual L&D budget?" (budget → Finance, answer lives in
   HR-007) and "How much tuition reimbursement can I claim?" (reimbursement → Finance,
   also HR-007). Encoded as `hr_ld_budget_wording` and `hr_tuition_reimbursement`.
   No keyword list fixes this — it needs the semantic router to arbitrate.
2. **Cross-department questions answer from one side.** `both_conference_travel_approval`
   ("Do I need approval to attend a conference that requires travel?") routes to Finance
   and returns the manager-approval travel rule, missing HR-007's requirement of VP
   approval for conference travel. Retrieval never sees the HR document.
3. **Upstream provider errors surface as HTTP 500.** *(fixed — see Next steps 4)* NVIDIA returns
   `503 ResourceExhausted: Worker local total request limit reached (107/32)` and
   `502 Bad Gateway` under sustained load; the API masks both as 500. The eval client
   retries 502/503/504 but not 500, so these become hard case failures. Cost 0 cases in
   run 3 and 5 cases in run 4 — the single largest source of run-to-run variance.
   Propagating upstream 5xx as 503 would let the existing retry path absorb them.
4. **The chat model occasionally emits raw scaffold.** Three answers in run 2 began
   "We need to answer based" followed by a run of `<unk>` tokens. Intermittent — the same
   cases passed in later runs. `nemotron-3-super-120b` is a reasoning model and its
   thinking trace leaks into the response.
5. **Ports 8000 and 8001 are occupied** on the dev machine — 8001 by an unrelated
   LLM-gateway service. Runs above used 8010.

### Scoring harness fixes

The first expanded run scored 22 failures; **13 were the scorer's fault, not the
system's.** The model emits typography that hand-written expected terms do not have:
U+202F narrow no-break space in "Net 30" and "November 30", U+2011 non-breaking hyphen
in "mid-size", U+2019 curly apostrophe in "don't have". `normalize()` in `run_eval.py`
now folds NFKC, quote/dash lookalikes, thousands separators, and `10 %` → `10%`, with
tests in `tests/test_run_eval.py`. Three of my own expected terms were also over-strict
(demanding "Workday" or "Expensify" when the answer was correct without naming the tool)
and were relaxed.

This is the reason to keep answers in `results.jsonl`: without reading them, a 51%
pass rate would have looked like a model problem.

## GraphRAG

Built on top of the vector index, not replacing it. `scripts/graph_index.py` runs
extraction -> entity resolution -> hierarchical communities -> community summaries;
`src/app/retrieval/graphrag.py` serves local and global search; `mode="graph"` on
`/ask` routes to it.

**What is in the graph:** 142 chunks -> 596 entities, 709 `RELATED` edges, 1045
`MENTIONS` edges, 3 community levels (174/103/95), 55 community summaries.
Hub entities are Finance (42), Manager (38), HR (34), VP (25), Workday (20),
CFO (18), Board (17).

Communities cross the department boundary, which was the point — `2-42` groups
Finance with Hiring Manager and Headcount Requisition, `2-14` groups Contractor
with Procurement Policy and Purchase Order. If Leiden had merely rediscovered the
HR/Finance split, the graph would add nothing over the existing per-department
indexes.

### Measured: 15 multi-hop cases, vector vs graph

| Metric | Vector (fast) | Graph | |
|---|---|---|---|
| Fully passed | 1/15 | **4/15** | stable across 3 runs |
| Answer quality (term checks) | 3/15 | **10/15** | 3.3x, after entity resolution |
| Source recall | 0.553 | **0.826** | |
| Source hits (strict superset) | 2/15 | **7-8/15** | |
| Routing match | 4/15 | 13/15 | **artifact — see below** |
| p50 latency | 5.94s | 9.14s | graph is slower |

Ranges, not single numbers, because the provider throttles: three graph runs answered
15/15, 12/15, and 14/15 cases, and a case lost to a 503 scores as a failure. Latency
p95 reaches 192s on runs where retries kick in. `fully_passed` was 4/15 in all three.

The regression suite is unaffected: 64/75 in fast mode after the graph work, against
62/75 before it, which is inside run-to-run variance.

**The routing number is not a win.** Graph mode returns `department_routed: "both"`
unconditionally because it does not use the department split, and most multi-hop
cases expect `"both"`. It scores 13/15 by construction, not by routing better. The
honest headline is answer quality 3/15 -> 9/15 and source recall 0.55 -> 0.83.

### What the build actually taught

1. **Global search lost to local search.** Map-reduce over community summaries
   answered the precise aggregation questions badly ("I don't have enough
   information" for the November 30 trainings) because summarization had already
   discarded the specifics. Local search, which walks entities into real chunk
   text, answered the same question perfectly. Microsoft's global search is built
   for thematic sensemaking; precise multi-document aggregation is local/DRIFT
   territory. Global search is implemented and kept, but `mode="graph"` routes to
   local search.
2. **Top-k was the whole ballgame, again.** Raising the entity budget from 12 to 32
   moved source recall 0.689 -> 0.834 and quality 7/15 -> 9/15 with no other change.
   The original failure was never "vector search is bad", it was "the answer lives
   in more documents than the window holds" — and a graph with a narrow k
   reproduces the same bug.
3. **The extraction cache was keyed on `elementId`.** Neo4j reassigns those on
   `ingest.py --reset`, so the 142-call cache would silently miss every entry
   exactly when re-ingesting. Now keyed on a hash of the chunk text; verified by
   rebuilding the graph with 0 extraction calls.
4. **Attribution needed a guard.** The first `community_sources` returned 70 of 75
   documents, because hub entities like "Manager" are mentioned by nearly every
   policy. That would have scored a perfect, meaningless source recall. It now
   requires a document to mention 3+ entities of a community.

### Corrections to earlier claims in this file

- **GDS is not usable on this Aura instance.** `SHOW PROCEDURES` lists 443 `gds.*`
  entries, but calling `gds.graph.project` fails asking for a `sessionId`: GDS on
  Aura is the separately provisioned Graph Analytics Serverless product, not an
  in-database library. Clustering runs client-side in networkx instead.
- **Louvain, not Leiden.** networkx 3.6 exposes `leiden_partitions` only as a
  dispatch API needing an external backend. `louvain_partitions` is natively
  implemented and gives the same dendrogram levels. `uv add leidenalg python-igraph`
  to switch. Still zero new dependencies today.

### Model constraints found the hard way

Four models were tried for the extraction and map steps, which need structured output:

| Model | Result |
|---|---|
| `nemotron-3-super-120b` (configured chat model) | returns `None` — reasoning trace breaks tool calling |
| `mistralai/mistral-nemotron` | 500 on tool calls; fine at plain text, so it does the reduce step |
| `meta/llama-3.3-70b-instruct` | 503 ResourceExhausted at `(20/16)` |
| `nvidia/llama-3.3-nemotron-super-49b` | read timeout |
| `meta/llama-3.1-8b-instruct` | works, ~8s/chunk, survives 142 sequential calls |

The same reasoning-trace defect has now broken ragas scoring, graph extraction, and
the global-search map step. It is the single most expensive recurring problem in
this project.

## RAGAS

Working and routine. `--ragas` on any run now scores faithfulness and answer relevancy
with near-full coverage, in about 10s per case.

**The judge model was the entire problem**, not ragas and not the prompt:

| Judge | Result |
|---|---|
| `nemotron-3-super-120b` (configured chat model) | 1/7 cases — reasoning trace breaks the structured-output parser |
| `mistralai/mistral-nemotron` | parses, but ~186s/job; 12 samples exceeded 30 min and timed out |
| `meta/llama-3.1-8b-instruct` | ~10s/case, near-full coverage — now the default |

The earlier conclusion that "faithfulness is impractical, make it a nightly job" was
wrong. It was a bad judge, and the fix was the same one that unblocked graph extraction:
use a small model that does structured output reliably. Five cases went from
unscoreable to 5/5 in 84 seconds.

Graph mode was silently unscoreable until now — the router returned `contexts: []`, so
ragas had nothing to judge against. `local_search` and `global_search` now return the
context they actually gave the model.

### Measured: same 15 multi-hop cases, vector vs graph

| Metric | Vector | Graph |
|---|---|---|
| **Faithfulness** | 0.648 (n=12/15) | **0.854 (n=12/15)** |
| **Answer relevancy** | 0.623 (n=14/15) | **0.903 (n=15/15)** |
| Answer quality (term checks) | 3/15 | **10/15** |
| Source recall | 0.555 | **0.826** |
| Fully passed | 1/15 | **4/15** |

Same questions, same judge, same corpus — the only variable is the retriever. This is
the attributable before/after the plan set out to produce.

Full 75-case factoid suite in fast mode, for reference: faithfulness 0.762 (n=70/73),
relevancy 0.677 (n=73/73), 65/75 passed, p50 2.96s. Not comparable to the table above,
which uses a different and much harder question set.

## Next steps

1. ~~Faithfulness is judge-bound~~ — **solved. See RAGAS below.** (Original diagnosis kept for the record:) The judge model was
   swapped to `mistralai/mistral-nemotron` (verified live, 0.7s on a single call,
   non-reasoning, no thinking trace to break ragas' parser). That fixed the *parsing*
   failure but not the cost, and the cost is now the blocker.

   Measured: at `--ragas-workers 4`, 3 samples took 13 min with 2 jobs hitting the 600s
   timeout. At `--ragas-workers 1`, a 10-sample run completed 9 of 20 jobs in 28 min —
   **~186s per job, serially, with one timeout**. Since a single judge call answers in
   0.7s, a 3-minute serial job is not rate-limit backoff: `Faithfulness` decomposes the
   answer into statements and then issues one NLI call per statement, so cost scales
   with claim count. Rate limiting (`ResourceExhausted`, the same throttling behind the
   API's 500s) makes it worse under concurrency but is not the floor. Lowering workers
   trades one failure mode for the other; neither setting makes 75 cases practical.

   Options, cheapest first: score a fixed 15-case subset rather than all 75; run
   `ResponseRelevancy` alone on every case (embedding-only, seconds) and faithfulness
   only on the subset; or move scoring to a nightly job (step 5) and stop treating it
   as something a developer waits on. `--ragas-workers` and `--ragas-timeout` are now
   flags so this is tunable without editing the script.
2. **Entity resolution done, with a modest payoff.** `--stage resolve` folds
   trailing qualifiers and possessives ("VP Approval" -> "VP", "Travel Policy" ->
   "Travel") and folds plurals only when the singular is itself an entity, so
   "IRS" does not become "IR". Leading words are never stripped: "VP of Sales"
   and "Hiring Manager" are different roles from "VP" and "Manager", and merging
   them would be a regression. Merged 25 nodes (596 -> 571 entities, singletons
   461 -> 430) and moved multi-hop answer quality from 8-9/15 to **10/15**.

   The remaining ceiling is not duplication. 430 of 571 entities are still
   mentioned by exactly one chunk, but most are genuinely single-mention
   (specific thresholds, one-off obligations), so further merging has little
   left to collect. `fully_passed` stays 4/15 because `source_hit` demands a
   strict superset of 4-9 expected documents.
3. **Reranker on vs. off is now lower priority.** The graph delta is the better
   before/after story and it is measured. Note the cost if still wanted: deep mode
   uses `nemotron-3-ultra-550b` and a local `BAAI/bge-reranker-large` (~1.5GB).
3. **Fix the two routing defects above** (Known broken 1 and 2), then re-measure. This
   is a real, attributable improvement: 71/75 → 74/75 if both land.
4. ~~Propagate upstream 5xx as 503~~ — **done.** `/ask` now maps provider
   `502/503/504/ResourceExhausted` to 503 so the eval client's retry path engages;
   genuine defects still surface as 500. Recovered 2 of 3 cases lost in one run.
5. **Nightly eval job, not a CI gate.** A 75-case run takes ~5 minutes functionally;
   with the judge it is far longer. Nightly writing `results.jsonl` is both practical
   and more credible than blocking PRs.

## Resume bullets

Now partly measured. Status of each original claim:

| Claim | Status |
|---|---|
| RAGAS faithfulness 64→91%, relevance 68→89%, n=100 | Replaced with measured numbers: faithfulness **0.648 → 0.854**, relevancy **0.623 → 0.903**, n=15, by swapping vector retrieval for the entity graph. Not 91%, and n=15 not 100 — but real and reproducible from `evals/`. |
| BM25 + dense + cross-encoder via RRF | Misdescribed. Hybrid fusion is Neo4j's Lucene fulltext index; RRF fuses multi-query variants. Reranker defaults off and is skipped in fast mode. |
| JWT role-scoped retrieval (HR/Finance/Admin) | Does not exist. Build it or drop it. |
| FastAPI streaming | No streaming endpoint. |
| CI gate on >5% faithfulness regression | CI runs ruff/mypy/pytest only. See next steps 5. |
| p95 < 2.5s over 500+ queries | **p50 is 2.33s, p95 is 8.17s at n=75.** The p50 claim survives; the p95 claim does not. |
| LangGraph | In `pyproject.toml`, unused. `router.py` is LangChain + `ThreadPoolExecutor`. |
| Neo4j knowledge graph | **Now true.** 571 entities, 697 relationships, 3 Leiden/Louvain community levels, 117 community summaries, built by `scripts/graph_index.py`. Was a vector store with no relationships. |

What is defensible today:

*"Built a GraphRAG layer over a 75-document policy corpus (Neo4j entity graph, hierarchical
community detection, local/global search). On 15 multi-hop questions whose answers span
4-19 documents, RAGAS faithfulness improved 0.65 → 0.85 and answer relevancy 0.62 → 0.90
against the same corpus and judge."*

*"Grew the eval suite from 8 to 90 cases; it immediately exposed two router defects that
were refusing 25% of answerable questions."*

Both survive someone opening the repo: the before/after is in `evals/results_multihop_*.jsonl`,
the fixes are in `router.py`, and the index is reproducible from `scripts/graph_index.py`.

## Commands

```bash
# Ingest (first run; add --reset to replace an existing graph)
uv run python scripts/ingest.py

# Functional eval, 75 cases (~5 min)
uv run python scripts/run_eval.py --api-url http://127.0.0.1:8010 --mode fast \
  --timeout 180 --output evals/results_fast.jsonl

# With quality scoring. Keep --ragas-workers at 1: the judge rate-limits, and ragas
# answers a 429 by retrying until the per-job timeout, turning throughput into NaNs.
uv run python scripts/run_eval.py --api-url http://127.0.0.1:8010 --mode fast \
  --timeout 180 --ragas --ragas-workers 1

# Triage a run: which failures are real, which are the scorer being brittle
uv run python -c "
import json
for r in map(json.loads, open('evals/results_fast.jsonl')):
    if not r.get('passed'):
        print(r['id'], r.get('missing_required_terms'), repr(r.get('answer','')[:100]))
"
```
