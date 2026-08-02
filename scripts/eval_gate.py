"""Compare an eval run against a frozen baseline, and fail only on real regressions.

    uv run python scripts/eval_gate.py freeze evals/results_fast.jsonl --output evals/baseline_fast.json
    uv run python scripts/eval_gate.py check  evals/results_fast.jsonl --baseline evals/baseline_fast.json

**Why not "fail if pass rate drops 5%".** At n=75 a single case is 1.3 points, so a
5-point threshold fires on 4 cases flipping — which is well inside the run-to-run noise
this project has already measured: provider 503s alone have cost 0, 3, and 5 cases on
otherwise identical runs. An arbitrary threshold at this sample size produces a gate that
cries wolf until someone disables it, which is worse than no gate.

So the gate is statistical, and it exploits something most implementations throw away:
**the two runs are paired.** Every case is the same question against the same corpus, so
the right test is McNemar's on the cases that *changed verdict*, not a two-sample
comparison of aggregate rates. Ten cases flipping the same direction is decisive; ten
flipping each way is noise. A two-proportion test cannot tell those apart — it sees the
same aggregate rate either way.

Both are reported:

* **McNemar (exact, binomial)** decides. It conditions on the discordant pairs, which is
  where the information actually is.
* **Newcombe's hybrid-score interval** on the difference is reported for effect size,
  because "-4pp, 95% CI [-12pp, +3pp]" tells a reader more than a p-value.

Continuous metrics (latency percentiles, RAGAS means) are reported but **not gated**.
With this sample size and a provider that throttles unpredictably, any threshold tight
enough to catch a real drift would fire constantly on noise. Reporting a number you
refuse to gate on is more honest than gating on one you do not trust.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Metrics that are per-case booleans, so they can be tested properly.
GATED_METRICS = ("passed", "department_match", "source_hit", "quality_match")

#: Reported for context, never gated. See the module docstring.
REPORTED_METRICS = ("source_recall", "latency_seconds")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves near 0 and 1 — exactly where a pass rate of 0.95 lives.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z / denominator * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def newcombe_difference(
    successes_a: int, total_a: int, successes_b: int, total_b: int, z: float = 1.96
) -> tuple[float, float]:
    """Hybrid-score interval for (rate_a - rate_b), per Newcombe (1998).

    Composed from each arm's Wilson interval rather than a pooled normal
    approximation, so it does not produce bounds outside [-1, 1] on small,
    lopsided samples.
    """
    if total_a == 0 or total_b == 0:
        return (0.0, 0.0)
    p_a, p_b = successes_a / total_a, successes_b / total_b
    lower_a, upper_a = wilson_interval(successes_a, total_a, z)
    lower_b, upper_b = wilson_interval(successes_b, total_b, z)
    delta = p_a - p_b
    lower = delta - math.sqrt((p_a - lower_a) ** 2 + (upper_b - p_b) ** 2)
    upper = delta + math.sqrt((upper_a - p_a) ** 2 + (p_b - lower_b) ** 2)
    return (lower, upper)


def mcnemar_exact_p(regressions: int, improvements: int) -> float:
    """One-sided exact McNemar p-value for "current is worse than baseline".

    Conditions on the discordant pairs only: cases that passed before and fail
    now (`regressions`) versus the reverse (`improvements`). Cases that did not
    change carry no information about whether anything changed, and including
    them is what makes an aggregate-rate comparison so insensitive here.

    Under the null the direction of each flip is a fair coin, so this is a
    one-sided binomial tail. Exact rather than chi-square because the discordant
    count is routinely under 10, where the chi-square approximation is poor.
    """
    n = regressions + improvements
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(regressions, n + 1))
    return tail / (2**n)


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def summarize(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    answered = [r for r in results.values() if r.get("ok")]
    summary: dict[str, Any] = {"n": len(results), "ok": len(answered)}
    for metric in GATED_METRICS:
        summary[metric] = sum(1 for r in results.values() if r.get(metric))
    recalls = [r["source_recall"] for r in answered if r.get("source_recall") is not None]
    summary["source_recall"] = round(sum(recalls) / len(recalls), 4) if recalls else None
    latencies = sorted(r["latency_seconds"] for r in answered if r.get("latency_seconds") is not None)
    if latencies:
        summary["p50_latency"] = latencies[max(math.ceil(0.50 * len(latencies)) - 1, 0)]
        summary["p95_latency"] = latencies[max(math.ceil(0.95 * len(latencies)) - 1, 0)]
    ragas: dict[str, list[float]] = {}
    for row in answered:
        for name, value in (row.get("ragas") or {}).items():
            ragas.setdefault(name, []).append(value)
    summary["ragas"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(ragas.items())}
    return summary


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - freezing a baseline outside a checkout is fine
        return "unknown"


def freeze(args: argparse.Namespace) -> int:
    results = load_results(args.results)
    baseline = {
        "created": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "source": str(args.results),
        "summary": summarize(results),
        # Per-case verdicts are the point: without them the comparison cannot be
        # paired, and an unpaired comparison at this n cannot detect anything.
        "cases": {
            case_id: {metric: bool(row.get(metric)) for metric in GATED_METRICS}
            for case_id, row in results.items()
        },
    }
    args.output.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"froze {len(results)} cases from {args.results} -> {args.output}")
    print(json.dumps(baseline["summary"], indent=2))
    return 0


def check(args: argparse.Namespace) -> int:
    baseline = json.loads(args.baseline.read_text())
    current = load_results(args.results)
    base_cases: dict[str, dict[str, bool]] = baseline["cases"]

    shared = sorted(set(base_cases) & set(current))
    added = sorted(set(current) - set(base_cases))
    dropped = sorted(set(base_cases) - set(current))

    lines: list[str] = []
    failures: list[str] = []
    lines.append(f"Baseline `{baseline['commit'][:8]}` ({baseline['created'][:10]}) vs current run")
    lines.append("")
    if added or dropped:
        lines.append(f"Suite changed: {len(added)} case(s) added, {len(dropped)} removed. "
                     f"Comparing the {len(shared)} shared cases.")
        lines.append("")

    lines.append("| metric | baseline | current | change | 95% CI on change | discordant | McNemar p |")
    lines.append("|---|---|---|---|---|---|---|")
    for metric in GATED_METRICS:
        base_pass = sum(1 for c in shared if base_cases[c].get(metric))
        curr_pass = sum(1 for c in shared if current[c].get(metric))
        regressions = sum(1 for c in shared if base_cases[c].get(metric) and not current[c].get(metric))
        improvements = sum(1 for c in shared if not base_cases[c].get(metric) and current[c].get(metric))
        low, high = newcombe_difference(curr_pass, len(shared), base_pass, len(shared))
        p_value = mcnemar_exact_p(regressions, improvements)
        delta = (curr_pass - base_pass) / len(shared) if shared else 0.0
        verdict = ""
        if p_value < args.alpha:
            verdict = " **REGRESSION**"
            failures.append(
                f"{metric}: {base_pass}/{len(shared)} -> {curr_pass}/{len(shared)} "
                f"({regressions} regressed, {improvements} improved, p={p_value:.4f})"
            )
        lines.append(
            f"| {metric} | {base_pass}/{len(shared)} | {curr_pass}/{len(shared)} | "
            f"{delta:+.1%} | [{low:+.1%}, {high:+.1%}] | {regressions}↓ {improvements}↑ | "
            f"{p_value:.3f}{verdict} |"
        )

    summary = summarize(current)
    base_summary = baseline["summary"]
    lines.append("")
    lines.append("Reported, not gated (sample size and provider variance make any threshold noise):")
    lines.append("")
    lines.append("| metric | baseline | current |")
    lines.append("|---|---|---|")
    for key in ("ok", "source_recall", "p50_latency", "p95_latency"):
        if key in summary or key in base_summary:
            lines.append(f"| {key} | {base_summary.get(key)} | {summary.get(key)} |")
    for name in sorted(set(summary.get("ragas", {})) | set(base_summary.get("ragas", {}))):
        lines.append(
            f"| ragas_{name} | {base_summary.get('ragas', {}).get(name)} | "
            f"{summary.get('ragas', {}).get(name)} |"
        )

    if regressed := [c for c in shared if base_cases[c].get("passed") and not current[c].get("passed")]:
        lines.append("")
        lines.append(f"Cases that regressed: {', '.join(f'`{c}`' for c in regressed[:12])}")

    report = "\n".join(lines)
    print(report)
    if args.markdown:
        args.markdown.write_text(report + "\n")

    if failures:
        print("\nGATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nGate passed: no change clears the noise floor at "
          f"alpha={args.alpha} (paired McNemar).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze and compare eval baselines.")
    sub = parser.add_subparsers(dest="command", required=True)

    freezer = sub.add_parser("freeze", help="Write a baseline from a results file.")
    freezer.add_argument("results", type=Path)
    freezer.add_argument("--output", type=Path, required=True)
    freezer.set_defaults(func=freeze)

    checker = sub.add_parser("check", help="Compare a results file against a baseline.")
    checker.add_argument("results", type=Path)
    checker.add_argument("--baseline", type=Path, required=True)
    checker.add_argument("--alpha", type=float, default=0.05)
    checker.add_argument("--markdown", type=Path, default=None, help="Also write the report here.")
    checker.set_defaults(func=check)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
