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


def ask(api_url: str, question: str, department: str | None) -> tuple[dict[str, Any], float]:
    payload = json.dumps({"question": question, "department": department}).encode("utf-8")
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=90) as response:
        elapsed = time.perf_counter() - started
        return json.loads(response.read().decode("utf-8")), elapsed


def source_ids(response: dict[str, Any]) -> set[str]:
    return {
        source["source"]
        for source in response.get("sources", [])
        if isinstance(source, dict) and source.get("source")
    }


def evaluate_case(case: dict[str, Any], api_url: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response, latency_seconds = ask(api_url, case["question"], case.get("department"))
        expected_sources = set(case.get("expected_sources", []))
        actual_sources = source_ids(response)
        return {
            "id": case["id"],
            "ok": True,
            "latency_seconds": round(latency_seconds, 3),
            "department_expected": case["expected_department"],
            "department_actual": response.get("department_routed"),
            "department_match": response.get("department_routed") == case["expected_department"],
            "expected_sources": sorted(expected_sources),
            "actual_sources": sorted(actual_sources),
            "source_hit": expected_sources.issubset(actual_sources),
            "answer": response.get("answer", ""),
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "id": case["id"],
            "ok": False,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BossAssistant API eval cases.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=Path("evals/questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evals/results.jsonl"))
    args = parser.parse_args()

    cases = load_cases(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = [evaluate_case(case, args.api_url) for case in cases]
    args.output.write_text("\n".join(json.dumps(result) for result in results) + "\n")

    passed = sum(1 for result in results if result.get("ok"))
    department_matches = sum(1 for result in results if result.get("department_match"))
    source_hits = sum(1 for result in results if result.get("source_hit"))
    avg_latency = sum(float(result["latency_seconds"]) for result in results) / len(results)

    print(f"cases={len(results)} ok={passed}")
    print(f"department_matches={department_matches}/{len(results)}")
    print(f"source_hits={source_hits}/{len(results)}")
    print(f"avg_latency_seconds={avg_latency:.2f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
