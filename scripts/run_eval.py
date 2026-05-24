from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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
) -> tuple[dict[str, Any], float, int]:
    payload = json.dumps({"question": question, "department": department, "mode": mode}).encode("utf-8")
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    last_exc: BaseException | None = None

    for attempt in range(1, retries + 2):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
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


def missing_required_terms(answer: str, required_terms: list[str]) -> list[str]:
    lowered = answer.lower()
    return [term for term in required_terms if term.lower() not in lowered]


def present_forbidden_terms(answer: str, forbidden_terms: list[str]) -> list[str]:
    lowered = answer.lower()
    return [term for term in forbidden_terms if term.lower() in lowered]


def evaluate_case(case: dict[str, Any], api_url: str, retries: int, retry_delay: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response, latency_seconds, attempts = ask(
            api_url,
            case["question"],
            case.get("department"),
            case.get("mode", "fast"),
            retries,
            retry_delay,
        )
        expected_sources = set(case.get("expected_sources", []))
        actual_sources = source_ids(response)
        answer = response.get("answer", "")
        missing_terms = missing_required_terms(answer, case.get("must_include", []))
        forbidden_terms = present_forbidden_terms(answer, case.get("must_not_include", []))
        department_match = response.get("department_routed") == case["expected_department"]
        source_hit = expected_sources.issubset(actual_sources)
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
            "missing_required_terms": missing_terms,
            "present_forbidden_terms": forbidden_terms,
            "quality_match": quality_match,
            "passed": department_match and source_hit and quality_match,
            "answer": answer,
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "id": case["id"],
            "ok": False,
            "attempts": retries + 1,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BossAssistant API eval cases.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=Path("evals/questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evals/results.jsonl"))
    parser.add_argument("--mode", choices=["fast", "deep"], default="fast")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = [
        evaluate_case({**case, "mode": args.mode}, args.api_url, args.retries, args.retry_delay) for case in cases
    ]
    args.output.write_text("\n".join(json.dumps(result) for result in results) + "\n")

    passed = sum(1 for result in results if result.get("ok"))
    department_matches = sum(1 for result in results if result.get("department_match"))
    source_hits = sum(1 for result in results if result.get("source_hit"))
    quality_matches = sum(1 for result in results if result.get("quality_match"))
    fully_passed = sum(1 for result in results if result.get("passed"))
    avg_latency = sum(float(result["latency_seconds"]) for result in results) / len(results)

    print(f"cases={len(results)} ok={passed}")
    print(f"department_matches={department_matches}/{len(results)}")
    print(f"source_hits={source_hits}/{len(results)}")
    print(f"quality_matches={quality_matches}/{len(results)}")
    print(f"fully_passed={fully_passed}/{len(results)}")
    print(f"avg_latency_seconds={avg_latency:.2f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
