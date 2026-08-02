from __future__ import annotations

import argparse
import json
import math
import os
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {502, 503, 504}
    return isinstance(exc, (TimeoutError, urllib.error.URLError))


def ask(
    api_url: str,
    question: str,
    department: str | None,
    mode: str,
    retries: int,
    retry_delay: float,
    timeout: float,
    include_contexts: bool = False,
    token: str | None = None,
) -> tuple[dict[str, Any], float, int]:
    payload = json.dumps(
        {
            "question": question,
            "department": department,
            "mode": mode,
            "include_contexts": include_contexts,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/ask",
        data=payload,
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    last_exc: BaseException | None = None

    for attempt in range(1, retries + 2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                elapsed = time.perf_counter() - started
                return json.loads(response.read().decode("utf-8")), elapsed, attempt
        except (TimeoutError, urllib.error.URLError) as exc:
            last_exc = exc
            if attempt > retries or not is_retryable_error(exc):
                raise
            time.sleep(retry_delay)

    raise RuntimeError("Request failed without an exception") from last_exc


def source_ids(response: dict[str, Any]) -> set[str]:
    return {
        source["source"]
        for source in response.get("sources", [])
        if isinstance(source, dict) and source.get("source")
    }


#: Typography the model emits that a hand-written expected term never has.
#: The answers really do come back with U+202F in "Net 30", U+2011 in
#: "mid-size", and U+2019 in "don't" — matching on raw ASCII scored a dozen
#: correct answers as misses.
_LOOKALIKES = str.maketrans(
    {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    }
)


def normalize(text: str) -> str:
    """Fold case, typography, and whitespace so a term check tests content.

    A model that writes "$2000" where the policy says "$2,000" is correct;
    without this the term check scores it as a miss.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_LOOKALIKES)
    collapsed = " ".join(folded.lower().replace(",", "").split())
    # NFKC turns the model's "10<U+202F>%" into "10 %"; the space is typography,
    # not content.
    return collapsed.replace(" %", "%")


def missing_required_terms(answer: str, required_terms: list[str]) -> list[str]:
    normalized = normalize(answer)
    return [term for term in required_terms if normalize(term) not in normalized]


def present_forbidden_terms(answer: str, forbidden_terms: list[str]) -> list[str]:
    normalized = normalize(answer)
    return [term for term in forbidden_terms if normalize(term) in normalized]


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile.

    `statistics.quantiles` interpolates and needs n>=2; a latency p95 wants
    an observed sample and has to survive a one-case run.
    """
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[max(rank - 1, 0)]


def evaluate_case(
    case: dict[str, Any],
    api_url: str,
    retries: int,
    retry_delay: float,
    timeout: float,
    include_contexts: bool = False,
    token: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response, latency_seconds, attempts = ask(
            api_url,
            case["question"],
            case.get("department"),
            case.get("mode", "fast"),
            retries,
            retry_delay,
            timeout,
            include_contexts,
            token,
        )
        expected_sources = set(case.get("expected_sources", []))
        actual_sources = source_ids(response)
        answer = response.get("answer", "")
        missing_terms = missing_required_terms(answer, case.get("must_include", []))
        forbidden_terms = present_forbidden_terms(answer, case.get("must_not_include", []))
        department_match = response.get("department_routed") == case["expected_department"]
        source_hit = expected_sources.issubset(actual_sources)
        # Aggregation questions expect 6-19 documents while retrieval returns 4,
        # so `source_hit` is all-or-nothing and pinned at False. Recall shows
        # movement — "found 2 of 11" and "found 9 of 11" are very different
        # answers that the boolean scores identically.
        source_recall = (
            len(expected_sources & actual_sources) / len(expected_sources)
            if expected_sources
            else None
        )
        quality_match = not missing_terms and not forbidden_terms
        return {
            "id": case["id"],
            "ok": True,
            "mode": case.get("mode", "fast"),
            "attempts": attempts,
            "latency_seconds": round(latency_seconds, 3),
            "department_expected": case["expected_department"],
            "department_actual": response.get("department_routed"),
            "department_match": department_match,
            "expected_sources": sorted(expected_sources),
            "actual_sources": sorted(actual_sources),
            "source_hit": source_hit,
            "source_recall": source_recall,
            "abstained": bool(response.get("abstained")),
            "expect_abstention": bool(case.get("expect_abstention")),
            "missing_required_terms": missing_terms,
            "present_forbidden_terms": forbidden_terms,
            "quality_match": quality_match,
            "passed": department_match and source_hit and quality_match,
            "question": case["question"],
            "answer": answer,
            "contexts": response.get("contexts") or [],
            "reference": case.get("reference"),
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "id": case["id"],
            "ok": False,
            "attempts": retries + 1,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": str(exc),
        }


def score_with_ragas(
    results: list[dict[str, Any]],
    judge_model: str,
    embed_model: str,
    workers: int = 1,
    timeout: int = 600,
) -> dict[str, tuple[float, int, int]]:
    """Score answered cases for faithfulness and answer relevance, in place.

    Imported lazily: ragas is a dev-only dependency with a heavy import
    chain, so a plain smoke run never pays for it.

    Only cases that came back with retrieved contexts are scored — the
    clarification cases retrieve nothing, and faithfulness against an empty
    context is undefined, not zero.
    """
    from langchain_core.embeddings import Embeddings
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig

    class SymmetricEmbeddings(Embeddings):
        """Embed both sides of a comparison with the query encoder.

        ResponseRelevancy compares the original question against questions it
        generated back from the answer — but it embeds the first with
        `embed_query` and the rest with `embed_documents`. On an asymmetric
        query/passage model (`nv-embedqa-e5-v5`) that scores two paraphrases of
        the same question as unrelated, which is how a passing answer lands at
        0.0. Both sides are questions, so both take the query encoder.

        Delegates rather than subclasses: NVIDIAEmbeddings validates the model
        name against the concrete client class, and a subclass fails that check.
        """

        def __init__(self, inner: NVIDIAEmbeddings) -> None:
            self._inner = inner

        def embed_query(self, text: str) -> list[float]:
            return self._inner.embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._inner._embed(texts, model_type="query")

    scorable = [result for result in results if result.get("ok") and result.get("contexts")]
    if not scorable:
        return {}

    # This script is a standalone HTTP client — it never builds app Settings,
    # so nothing else has read .env into the environment.
    load_dotenv()
    api_key = os.environ["NVIDIA_API_KEY"]
    llm = LangchainLLMWrapper(ChatNVIDIA(model=judge_model, api_key=api_key, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(
        SymmetricEmbeddings(NVIDIAEmbeddings(model=embed_model, api_key=api_key, truncate="END"))
    )

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=result["question"],
                response=result["answer"],
                retrieved_contexts=result["contexts"],
                reference=result["reference"],
            )
            for result in scorable
        ]
    )
    # `evaluate` is typed as returning EvaluationResult | Executor (it can hand
    # back an executor instead); we never ask for the executor form.
    evaluation: Any = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=llm,
        embeddings=embeddings,
        # Workers default to 1 because the hosted judge rate-limits and ragas
        # answers a 429 by retrying until the per-job timeout, turning extra
        # workers into NaN scores. Serial is not fast either: Faithfulness
        # issues one NLI call per extracted statement, measured at ~186s per
        # job even at one worker. Scoring the full set is a nightly job, not
        # something to wait on.
        run_config=RunConfig(timeout=timeout, max_workers=workers),
    )
    per_case = evaluation.scores

    totals: dict[str, list[float]] = {}
    for result, case_scores in zip(scorable, per_case, strict=True):
        scored = {
            name: round(float(value), 3)
            for name, value in case_scores.items()
            if isinstance(value, (int, float)) and not math.isnan(value)
        }
        result["ragas"] = scored
        for name, value in scored.items():
            totals.setdefault(name, []).append(value)

    for name, values in sorted(totals.items()):
        if len(values) < len(scorable):
            print(
                f"WARNING: {name} scored only {len(values)}/{len(scorable)} cases; "
                "the rest returned NaN (judge timeout or unparseable output). "
                "Treat the mean as provisional."
            )

    if not totals:
        # ragas swallows per-job exceptions (dead judge model, bad key, rate
        # limit) and returns NaN. Reporting "no metrics" quietly here is how a
        # broken eval gets mistaken for a passing one.
        raise RuntimeError(
            f"ragas produced no usable scores for {len(scorable)} case(s) — "
            "check the judge model is live and NVIDIA_API_KEY is valid"
        )

    # Coverage rides along with the mean: a metric averaged over 1 of 8 cases
    # must never be readable as a headline score.
    return {
        name: (sum(values) / len(values), len(values), len(scorable))
        for name, values in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BossAssistant API eval cases.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=Path("evals/questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evals/results.jsonl"))
    parser.add_argument("--mode", choices=["fast", "deep", "graph"], default="fast")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ragas", action="store_true", help="Score faithfulness and answer relevance.")
    # Judge choice is the whole ballgame for ragas throughput, not a detail.
    # nemotron-3-super-120b scored 1 of 7 cases (reasoning trace breaks the
    # structured-output parser); mistral-nemotron parsed but averaged ~186s per
    # job and timed out at scale. llama-3.1-8b does structured output reliably
    # and scores a case in ~10s, which is what turned faithfulness from
    # unmeasurable into a routine part of the run.
    parser.add_argument("--ragas-model", default="meta/llama-3.1-8b-instruct")
    parser.add_argument("--ragas-embed-model", default="nvidia/nv-embedqa-e5-v5")
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token for the ACL-filtered /ask endpoint. Baselines use a token "
        "holding every group, so the numbers measure retrieval quality rather than "
        "ACL selectivity; scripts/measure_acl_recall.py covers the ACL cost separately.",
    )
    parser.add_argument("--ragas-workers", type=int, default=2)
    parser.add_argument("--ragas-timeout", type=int, default=600)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = [
        evaluate_case(
            {**case, "mode": args.mode},
            args.api_url,
            args.retries,
            args.retry_delay,
            args.timeout,
            include_contexts=args.ragas,
            token=args.token,
        )
        for case in cases
    ]

    def write_results() -> None:
        args.output.write_text("\n".join(json.dumps(result) for result in results) + "\n")

    # Persist before scoring: the request data cost minutes of live API time,
    # and a judge failure must not discard it.
    write_results()
    if args.ragas:
        ragas_means = score_with_ragas(
            results,
            args.ragas_model,
            args.ragas_embed_model,
            args.ragas_workers,
            args.ragas_timeout,
        )
        write_results()
    else:
        ragas_means = {}

    passed = sum(1 for result in results if result.get("ok"))
    department_matches = sum(1 for result in results if result.get("department_match"))
    source_hits = sum(1 for result in results if result.get("source_hit"))
    quality_matches = sum(1 for result in results if result.get("quality_match"))
    fully_passed = sum(1 for result in results if result.get("passed"))
    # Abstention is scored separately from correctness. A system that abstains on
    # everything scores perfectly on faithfulness and is useless, so precision
    # (was the refusal warranted?) and recall (did it refuse when it should?) are
    # reported as a pair — neither is meaningful alone.
    abstained = [r for r in results if r.get("abstained")]
    should = [r for r in results if r.get("expect_abstention")]
    correct_abstentions = [r for r in abstained if r.get("expect_abstention")]
    recalls = [
        float(result["source_recall"])
        for result in results
        if result.get("source_recall") is not None
    ]
    latencies = [float(result["latency_seconds"]) for result in results]
    avg_latency = sum(latencies) / len(latencies)

    print(f"cases={len(results)} ok={passed}")
    print(f"department_matches={department_matches}/{len(results)}")
    print(f"source_hits={source_hits}/{len(results)}")
    print(f"quality_matches={quality_matches}/{len(results)}")
    print(f"fully_passed={fully_passed}/{len(results)}")
    if recalls:
        print(f"source_recall={sum(recalls) / len(recalls):.3f} (n={len(recalls)})")
    if abstained or should:
        precision = len(correct_abstentions) / len(abstained) if abstained else float("nan")
        recall_rate = len(correct_abstentions) / len(should) if should else float("nan")
        print(
            f"abstention_precision={precision:.3f} ({len(correct_abstentions)}/{len(abstained)}) "
            f"abstention_recall={recall_rate:.3f} ({len(correct_abstentions)}/{len(should)})"
        )
    print(f"avg_latency_seconds={avg_latency:.2f}")
    print(f"p50_latency_seconds={percentile(latencies, 50):.2f}")
    print(f"p95_latency_seconds={percentile(latencies, 95):.2f}")
    for name, (mean, scored_n, total_n) in sorted(ragas_means.items()):
        print(f"ragas_{name}={mean:.3f} (n={scored_n}/{total_n})")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
