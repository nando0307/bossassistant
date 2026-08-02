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
| Source recall | 0.553 | **disputed — see note** | |
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

**Re-measured after the source-attribution fix. The earlier recall figure was wrong and
the correction reverses that claim.**

| Metric | Vector | Graph |
|---|---|---|
| Answer quality (term checks) | 3/15 | **7/15** |
| Source recall | **0.546** | **0.435** |
| Source hits (strict superset) | 2/15 | 2/15 |
| Fully passed | 1/15 | 1/15 |
| p50 latency | 5.33s | 7.37s |

What survives: **graph mode answers roughly twice as many multi-hop questions correctly**
(3/15 -> 7/15). Its advantage is assembling an answer whose evidence spans more documents
than a top-k window holds.

What does not survive: the claim that graph mode improves *source recall*. It was
reported as 0.553 -> 0.826, and that number was an artifact of the dragnet described
above — local search credited every document mentioning any one of its 32 matched
entities, citing a median of 27 of 75 policies per answer, which makes hitting the
expected source nearly free. With attribution requiring 3+ entity hits, graph recall is
**0.435, below vector's 0.546**. Graph mode finds the right answer while citing fewer of
the documents the case expects.

That is a worse result than previously reported and it is the correct one. The retrieval
claim to make is about answer completeness on multi-document questions, not about recall.

RAGAS scores are absent from this re-baseline: the judge was throttling badly (30+ minutes
on a single 75-case scoring pass versus ~7 minutes when first measured), so baselines were
frozen without it. The gate does not use RAGAS — it gates on per-case booleans and reports
RAGAS only as context — so this blocks nothing, but the 0.648 -> 0.854 faithfulness figure
predates the attribution fix and should be re-run before it is quoted.

## ACL-aware retrieval

Retrieval is filtered by the caller's JWT groups. The groups are pushed **down into
`retrieve()`**, not checked against the finished answer — by the time an answer exists,
restricted text has already entered the prompt and can leak through paraphrase, refusal
wording, or a citation list.

**The ACL axis is deliberately not the department axis.** 14 of 75 documents are
restricted: compensation bands and PIPs to `managers`/`hr-team`, SOX/treasury/transfer
pricing to `finance-team`/`executives`, M&A to `executives` alone. If ACL collapsed onto
department, the existing per-department indexes would already be the whole feature.

Verified end to end on "What approvals are required for an acquisition?":

| Caller | Sources returned | Answer |
|---|---|---|
| no token | — | 401 `missing bearer token` |
| `all-employees` | FIN-008, FIN-014, FIN-013, FIN-033 | "I don't have that information" |
| `executives` | **FIN-037**, FIN-008, FIN-014 | cites the M&A policy |

Graph mode is filtered too, and needed an extra guard: an `Entity.description` is
LLM-written *from* the source text, so an entity extracted only from a restricted policy
discloses that policy in paraphrase even if the chunk itself is withheld. Entities are
therefore only visible when the caller can read a chunk that mentions them.

`global_search` is a different matter and **refuses ACL-scoped calls rather than
under-enforcing**. Its community summaries are written once by an LLM across the whole
corpus, so a summary of the "board approvals" cluster paraphrases the M&A policy whether
or not the reader may open it; filtering only its sources would return an answer that
looks scoped while its prose is not. Making it ACL-aware means summarizing per access
tier at index time, deferred until global search is actually on a request path —
`mode="graph"` routes to `local_search`, which filters properly.

> A note on how this was found. The first version of the graph filter **silently did
> nothing**: `local_search` accepted a `groups` argument, but the edit to its Cypher
> never applied, so the parameter was ignored and an `all-employees` token retrieved the
> executives-only M&A policy. A security control that is present in the signature and
> absent from the query looks identical to a working one in code review. There is now a
> test asserting `$groups` and `acl_groups` actually appear in the executed statement.

### The trade-off, measured

Neo4j's vector index **cannot pre-filter**: `db.index.vector.queryNodes` picks ANN
neighbours first and any property predicate applies afterwards. So the filter is
necessarily post-ANN, and the failure is silent — a narrowly-scoped caller gets fewer
than `k` chunks, or none, while the query looks successful.

Two options: over-fetch `k*f` then filter (one index, recall degrades with ACL
selectivity), or partition per group (exact, but combinatorial once groups overlap — a
user in 3 of 5 groups needs a union of 3 indexes, and a document in N groups is embedded
N times). This project takes over-fetch. `scripts/measure_acl_recall.py` prices it
against exact brute-force cosine over every readable chunk:

| Persona | readable | f=1 | f=2 | f=4 | f=6 | f=12 |
|---|---|---|---|---|---|---|
| employee | 89% / 72% | 0.880 | 0.970 | 0.970 | 0.970 | 0.970 |
| finance-analyst | 89% / 97% | 0.880 | 0.970 | 0.970 | 0.970 | 0.970 |
| hr-partner | 100% / 72% | 0.970 | 0.970 | 0.970 | 0.970 | 0.970 |
| executive | 89% / 100% | 0.880 | 0.970 | 0.970 | 0.970 | 0.970 |
| **executive-only** | **0% / 17%** | **0.000** | **0.000** | 0.625 | **1.000** | 1.000 |

Two things this shows that an opinion would not:

1. **Selectivity, not group count, drives the cost.** Broad personas are saturated by
   f=2. The principal who can read 17% of Finance and 0% of HR scores **0.000 recall at
   f=2 and returns empty on 2 of 25 questions with no error at all.** That is the silent
   degradation, reproduced.
2. **`ACL_OVERFETCH = 6` is chosen, not guessed** — the first factor where the most
   selective principal reaches 1.000.

The 0.970 ceiling is not an ACL cost: it persists at f=12 and is the hybrid ANN index's
own approximation error against exact cosine.

## Ingestion that survives a changing corpus

`scripts/ingest.py` was a one-shot create-only script: re-running it either duplicated
every chunk or demanded `--reset`, and `--reset` destroys the GraphRAG entity graph built
on top of those nodes. It is now incremental.

Chunks have a stable identity — `(source, chunk_idx)` — plus a `content_hash`:

| state | action |
|---|---|
| unchanged | skipped entirely: no embedding call, no write |
| changed | re-embedded, updated in place |
| new | created |
| no longer produced by the corpus | **deleted** |

**Idempotency is asserted, not claimed.** Second run over an unchanged corpus:
`146 unchanged, 0 writes, 0 embedding calls`. `tests/test_ingest.py` pins it with a fake
graph so it keeps running without Aura credentials.

Three subtleties that cost real incidents:

1. **`chunk_idx` is per-document, not per-department.** Splitting a whole department at
   once makes every index depend on how many chunks preceded it, so a one-word edit near
   the top marks the entire corpus changed and re-embeds all of it.
2. **Shortened documents orphan their tail.** A document that used to split into 4 chunks
   and now splits into 2 leaves chunks 2 and 3 behind — still retrievable, still cited.
3. **Legacy nodes migrate themselves.** The old ingester wrote no `chunk_idx`. Filtering
   those out of the diff would have hidden them and written a second full set alongside;
   treating them as unmatched puts them on the delete path. The dry run caught this
   before it ran: `new 73, deleted 0` on the first attempt, which would have left 288
   chunks where 146 belong.

Deletes use `DETACH DELETE` because these chunks carry `MENTIONS` edges into the entity
graph — leaving the relationships would keep a withdrawn policy reachable through
entities even after its chunk is gone.

### Supersession

Documents carry `effective_date`, and a revision declares `supersedes`; the reverse
`superseded_by` pointer is derived so there is one place to edit when a policy is
replaced. Superseded chunks are dropped **at retrieval**, in both vector and graph modes,
so replaced text never enters the prompt.

Two real revisions exercise it: HR-039 replaces HR-016 (jury duty 10 -> 15 days) and
FIN-038 replaces FIN-020 (insurance renewal July 1 -> September 1). Both were chosen
because no eval case cites them, so supersession is demonstrable without rewriting the
suite. Verified live on "How many days of paid jury duty leave do I get per year?":

| mode | sources | answer |
|---|---|---|
| fast | HR-039, HR-001, HR-017, HR-015 | "15 business days per calendar year (Jury Duty and Civic Leave, **effective 2026-07-01**)" |
| graph | HR-039 (+ others), **no HR-016** | "15 business days" |

Answers now carry the vintage of the policy they quote, because a policy figure without
an effective date cannot be verified by the person reading it.

**Not done: layout-aware parsing of real PDFs.** The corpus is still clean synthetic
prose, which does not exercise table extraction, multi-column layout, or headers bleeding
into chunks. This needs real source documents rather than better code — generating PDFs
from this same text would test nothing. Public HR/finance policy PDFs or SEC filings are
the realistic input.

## The CI eval gate

Nightly, not per-PR: a scored run is ~90 live LLM calls plus a judge pass per case, and
the regression signal is a property of the model and the provider — which drift on their
own schedule, not per commit. `.github/workflows/nightly-eval.yml` runs the suite, diffs
against a frozen baseline, writes a job summary, uploads results, and opens an issue
(labelled `eval-regression`) only when the gate actually fails.

**Why the gate is not "fail if the pass rate drops 5%".** At n=75 a single case is 1.3
points, so a 5-point threshold fires on 4 cases flipping — well inside noise this project
has already measured: provider 503s alone cost 0, 3, and 5 cases on otherwise identical
runs. An arbitrary threshold at this sample size produces a gate that cries wolf until
someone switches it off, which is worse than no gate at all.

**The runs are paired**, and that is the part most implementations discard. Every case is
the same question against the same corpus, so the information lives in the cases that
*changed verdict*, not in the aggregate rate. The gate uses an exact one-sided **McNemar**
test on discordant pairs, and reports a **Newcombe hybrid-score interval** on the
difference for effect size.

The re-baseline produced a clean demonstration of why that matters. Comparing graph mode
against the fast-mode baseline:

| metric | baseline | current | change | 95% CI | discordant | McNemar p | |
|---|---|---|---|---|---|---|---|
| passed | 64/75 | 18/75 | -61.3% | [-71.7%, -46.8%] | 47↓ 1↑ | 0.000 | **REGRESSION** |
| department_match | 72/75 | 30/75 | -56.0% | [-66.7%, -42.6%] | 43↓ 1↑ | 0.000 | **REGRESSION** |
| source_hit | 68/75 | 51/75 | -22.7% | [-34.8%, -9.8%] | 17↓ 0↑ | 0.000 | **REGRESSION** |
| quality_match | 67/75 | 67/75 | +0.0% | [-10.4%, +10.4%] | **7↓ 7↑** | 0.605 | not flagged |

`quality_match` is the case that makes the argument: **identical aggregate rate, 14 cases
changed verdict, correctly not flagged.** A two-proportion test sees 67/75 both times and
cannot distinguish that from nothing happening — and equally, it would miss 10 cases
regressing if 10 others improved. Pairing separates them.

Continuous metrics (latency percentiles, RAGAS means) are reported but deliberately **not
gated**: with this sample size and a provider that throttles unpredictably, any threshold
tight enough to catch real drift would fire constantly on noise. Reporting a number you
refuse to gate on is more honest than gating on one you do not trust.

Frozen baselines live at `evals/baseline_fast.json` and `evals/baseline_graph.json`, and
carry per-case verdicts — without them the comparison cannot be paired, and an unpaired
comparison at this n cannot detect much of anything.

**Not done: the production feedback loop.** Capturing thumbs-down plus trace ID from real
traffic, triaging weekly, and promoting genuine failures into `evals/` is what separates
an eval *suite* from an eval *process*. It needs a `/feedback` endpoint and a frontend
control, neither of which exists yet.

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
5. ~~Nightly eval job~~ — **done, see "The CI eval gate" above.** (Original note:) A 75-case run takes ~5 minutes functionally;
   with the judge it is far longer. Nightly writing `results.jsonl` is both practical
   and more credible than blocking PRs.

## Resume bullets

Now partly measured. Status of each original claim:

| Claim | Status |
|---|---|
| RAGAS faithfulness 64→91%, relevance 68→89%, n=100 | Faithfulness 0.648 → 0.854 and relevancy 0.623 → 0.903 were measured at n=15, but **predate the source-attribution fix** and need re-running before use. The durable claim is answer quality on multi-hop questions: **3/15 → 7/15**. |
| BM25 + dense + cross-encoder via RRF | Misdescribed. Hybrid fusion is Neo4j's Lucene fulltext index; RRF fuses multi-query variants. Reranker defaults off and is skipped in fast mode. |
| JWT role-scoped retrieval (HR/Finance/Admin) | **Now real**, and more interesting than the original claim: 5 groups that cut across department, filtered at retrieval rather than post-hoc, with the post-ANN recall cost measured (`evals/acl_recall.json`). |
| FastAPI streaming | No streaming endpoint. |
| CI gate on >5% faithfulness regression | CI runs ruff/mypy/pytest only. See next steps 5. |
| p95 < 2.5s over 500+ queries | **p50 is 2.33s, p95 is 8.17s at n=75.** The p50 claim survives; the p95 claim does not. |
| LangGraph | In `pyproject.toml`, unused. `router.py` is LangChain + `ThreadPoolExecutor`. |
| Neo4j knowledge graph | **Now true.** 571 entities, 697 relationships, 3 Leiden/Louvain community levels, 117 community summaries, built by `scripts/graph_index.py`. Was a vector store with no relationships. |

What is defensible today:

*"Built a GraphRAG layer over a 77-document policy corpus (Neo4j entity graph, hierarchical
community detection, local/global search). On 15 multi-hop questions whose answers span
4-19 documents, it answered 7/15 correctly against vector retrieval's 3/15 — while
measuring that it does **not** improve source recall (0.435 vs 0.546), because the first
version's apparent recall gain came from citing a third of the corpus per answer."*

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
