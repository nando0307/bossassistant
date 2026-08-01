"""Measure what ACL filtering costs recall, as a function of over-fetch.

Neo4j's vector index cannot pre-filter. `db.index.vector.queryNodes` selects
ANN neighbours first and any property predicate is applied to that result set,
so an ACL filter is necessarily post-ANN. The failure mode is silent: a caller
whose groups own a thin slice of the corpus gets back fewer than `k` chunks —
or none — while the query itself looks perfectly successful.

The two ways out:

1. **Over-fetch `k * f`, then filter.** One index, no write amplification, but
   recall degrades with the caller's ACL selectivity and you pay latency for
   candidates you throw away.
2. **Partition the index per group.** Exact recall, but the partition count is
   combinatorial once groups overlap — a user in 3 of 5 groups needs the union
   of 3 indexes, and every document in N groups is embedded N times.

This project takes option 1. This script measures the price so the choice is
defensible with a number rather than an opinion.

Ground truth is exact: the corpus is small enough to brute-force cosine
similarity over every readable chunk, which is what the ANN index is
approximating. Recall@k is measured against that.

    uv run python scripts/measure_acl_recall.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from app.retrieval.rag import INDEX_CONFIG, Department, get_embedder, get_vector_store

#: Realistic group memberships. The last one is deliberately pathological: a
#: principal holding only `executives` can read 2 of 142 chunks, which is where
#: post-ANN filtering breaks down.
PERSONAS: dict[str, frozenset[str]] = {
    "employee": frozenset({"all-employees"}),
    "finance-analyst": frozenset({"all-employees", "finance-team"}),
    "hr-partner": frozenset({"all-employees", "hr-team", "managers"}),
    "executive": frozenset({"all-employees", "executives", "finance-team"}),
    "executive-only": frozenset({"executives"}),
}


def load_chunks(graph_query: Any, department: Department) -> list[dict[str, Any]]:
    label = INDEX_CONFIG[department]["node_label"]
    return graph_query(
        f"MATCH (c:{label}) "
        "RETURN c.source AS source, c.text AS text, c.embedding AS embedding, "
        "coalesce(c.acl_groups, []) AS acl_groups"
    )


def true_top_k(
    chunks: list[dict[str, Any]], query_vector: np.ndarray, groups: frozenset[str], k: int
) -> list[str]:
    """Exact top-k over every chunk the principal may read."""
    readable = [c for c in chunks if groups.intersection(c["acl_groups"])]
    if not readable:
        return []
    matrix = np.array([c["embedding"] for c in readable], dtype=float)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    scores = matrix @ query_vector
    order = np.argsort(-scores)[:k]
    return [f"{readable[i]['source']}::{hash(readable[i]['text']) & 0xFFFF}" for i in order]


def measured_top_k(
    store: Any, question: str, groups: frozenset[str], k: int, overfetch: int
) -> list[str]:
    """What the production path returns: ANN over-fetch, then filter."""
    docs = store.similarity_search(question, k=k * overfetch)
    kept = [d for d in docs if groups.intersection(d.metadata.get("acl_groups") or [])]
    return [
        f"{d.metadata.get('source')}::{hash(d.page_content.removeprefix(chr(10) + 'text: ').strip()) & 0xFFFF}"
        for d in kept[:k]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure ACL over-fetch recall cost.")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12])
    parser.add_argument("--questions", type=Path, default=Path("evals/questions.jsonl"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("evals/acl_recall.json"))
    args = parser.parse_args()

    load_dotenv()
    from langchain_neo4j import Neo4jGraph

    from app.config import settings

    graph = Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )

    cases = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    cases = cases[: args.limit]
    embedder = get_embedder()

    departments: list[Department] = ["hr", "finance"]
    chunks = {d: load_chunks(graph.query, d) for d in departments}
    stores = {d: get_vector_store(d) for d in departments}
    for dept_name, rows in chunks.items():
        total = len(rows)
        for name, groups in PERSONAS.items():
            readable = sum(1 for c in rows if groups.intersection(c["acl_groups"]))
            print(f"  {dept_name}/{name}: {readable}/{total} chunks readable ({readable / total:.0%})")

    report: dict[str, Any] = {"k": args.k, "cases": len(cases), "personas": {}}
    print(f"\nmeasuring recall@{args.k} over {len(cases)} questions\n")
    header = "persona".ljust(17) + "".join(f"f={f}".rjust(9) for f in args.factors) + "   empty@f=1"
    print(header)
    print("-" * len(header))

    for name, groups in PERSONAS.items():
        row: dict[int, float] = {}
        empties = 0
        for factor in args.factors:
            recalls = []
            for case in cases:
                department: Department = "finance" if case["expected_department"] == "finance" else "hr"
                query_vector = np.array(embedder.embed_query(case["question"]), dtype=float)
                query_vector /= np.linalg.norm(query_vector)
                truth = true_top_k(chunks[department], query_vector, groups, args.k)
                if not truth:
                    continue
                got = measured_top_k(stores[department], case["question"], groups, args.k, factor)
                if factor == args.factors[0] and not got:
                    empties += 1
                recalls.append(len(set(got) & set(truth)) / len(truth))
            row[factor] = sum(recalls) / len(recalls) if recalls else float("nan")
        report["personas"][name] = {"groups": sorted(groups), "recall_by_factor": row, "empty_at_min_factor": empties}
        print(name.ljust(17) + "".join(f"{row[f]:.3f}".rjust(9) for f in args.factors) + f"   {empties}/{len(cases)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
