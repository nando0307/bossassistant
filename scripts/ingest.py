"""Load the HR and Finance policy corpus into Neo4j, incrementally.

The corpus itself lives in `app.corpus` so the vector ingester and the GraphRAG
indexer cannot drift apart. Index names, node labels, and the embedder come from
the retrieval module so ingestion and query time cannot drift apart either.

**Why this is not `Neo4jVector.from_documents`.** That helper creates fresh
nodes on every call, which is fine exactly once. A policy corpus changes: a
document gets revised, another is withdrawn, a third is re-worded. Re-running a
create-only ingester either duplicates every chunk or forces a full wipe, and a
full wipe throws away the GraphRAG entity graph built on top of these nodes.

So chunks get a stable identity — `(source, chunk_idx)` — plus a `content_hash`:

* unchanged chunk  -> skipped entirely, no embedding call, no write
* changed chunk    -> re-embedded and updated in place
* new chunk        -> created
* chunk no longer produced by the corpus -> **deleted**

That last case is the one enterprises actually get bitten by. A withdrawn policy
whose chunks linger is not a stale cache, it is an assistant citing a rule that
no longer exists. Shortened documents hit the same path: if a document used to
split into 4 chunks and now splits into 2, chunks 2 and 3 are orphans.

Re-running with no corpus change writes nothing. `tests/test_ingest.py` asserts
that, because "idempotent" is a claim that rots silently.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from typing import Any

from langchain_neo4j import Neo4jGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.corpus import DOCUMENTS
from app.retrieval.rag import INDEX_CONFIG, Department, get_embedder

#: nv-embedqa-e5-v5. Declared here because the vector index must be created with
#: the right width before any node is written.
EMBEDDING_DIMENSIONS = 1024


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit, with a stable identity across re-ingests."""

    source: str
    chunk_idx: int
    text: str
    content_hash: str
    metadata: dict[str, Any]

    @property
    def key(self) -> tuple[str, int]:
        return (self.source, self.chunk_idx)


def connect() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )


def build_chunks(docs: list[Any], splitter: RecursiveCharacterTextSplitter) -> list[Chunk]:
    """Split each document independently so `chunk_idx` is stable.

    Splitting the whole department at once would make every chunk's index depend
    on how many chunks the documents before it produced, so editing one document
    would renumber — and therefore appear to change — every chunk after it.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        for index, text in enumerate(splitter.split_text(doc.page_content)):
            chunks.append(
                Chunk(
                    source=doc.metadata["source"],
                    chunk_idx=index,
                    text=text,
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                    metadata=doc.metadata,
                )
            )
    return chunks


def ensure_indexes(graph: Neo4jGraph, department: Department) -> None:
    """Create the vector and fulltext indexes if they do not exist.

    `from_documents` used to do this as a side effect of writing nodes; writing
    nodes directly means owning it explicitly.
    """
    config = INDEX_CONFIG[department]
    label = config["node_label"]
    graph.query(
        f"CREATE VECTOR INDEX {config['index_name']} IF NOT EXISTS "
        f"FOR (c:{label}) ON (c.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: $dims, "
        "`vector.similarity_function`: 'cosine'}}",
        {"dims": EMBEDDING_DIMENSIONS},
    )
    graph.query(
        f"CREATE FULLTEXT INDEX {config['keyword_index_name']} IF NOT EXISTS "
        f"FOR (c:{label}) ON EACH [c.text]"
    )


def existing_chunks(graph: Neo4jGraph, department: Department) -> list[dict[str, Any]]:
    """Every node under this label, keyed by element id.

    Deliberately does *not* filter to nodes that already have `chunk_idx`.
    Nodes written by the old create-only ingester have none, so filtering them
    out would hide them from the diff and the next run would write a duplicate
    set alongside them. Treated as unmatched, they fall into the orphan path and
    are replaced — the migration happens by itself.
    """
    label = INDEX_CONFIG[department]["node_label"]
    return graph.query(
        f"MATCH (c:{label}) RETURN elementId(c) AS element_id, c.source AS source, "
        "c.chunk_idx AS chunk_idx, coalesce(c.content_hash, '') AS content_hash"
    )


def sync_department(
    graph: Neo4jGraph,
    department: Department,
    chunks: list[Chunk],
    dry_run: bool = False,
    embedder: Any | None = None,
) -> dict[str, int]:
    """Bring one department's chunks in line with the corpus. Returns a diff.

    `embedder` is injected so the diff logic can be tested without a live
    endpoint; production passes None and gets the configured embedder.
    """
    label = INDEX_CONFIG[department]["node_label"]
    ensure_indexes(graph, department)

    rows = existing_chunks(graph, department)
    wanted = {chunk.key: chunk for chunk in chunks}
    before = {
        (row["source"], row["chunk_idx"]): row["content_hash"]
        for row in rows
        if row["source"] is not None and row["chunk_idx"] is not None
    }

    upserts = [c for c in chunks if before.get(c.key) != c.content_hash]
    unchanged = len(chunks) - len(upserts)
    orphans = [
        row["element_id"]
        for row in rows
        if (row["source"], row["chunk_idx"]) not in wanted
    ]

    stats = {
        "total": len(chunks),
        "unchanged": unchanged,
        "new": sum(1 for c in upserts if c.key not in before),
        "changed": sum(1 for c in upserts if c.key in before),
        "deleted": len(orphans),
    }
    if dry_run:
        return stats

    if upserts:
        # Embedding is the expensive step; only unchanged text avoids it.
        vectors = (embedder or get_embedder()).embed_documents([c.text for c in upserts])
        payload = [
            {
                "source": c.source,
                "chunk_idx": c.chunk_idx,
                "text": c.text,
                "content_hash": c.content_hash,
                "embedding": vector,
                "title": c.metadata.get("title"),
                "department": c.metadata.get("department"),
                "acl_groups": list(c.metadata.get("acl_groups") or []),
                "effective_date": c.metadata.get("effective_date"),
                "superseded_by": c.metadata.get("superseded_by"),
            }
            for c, vector in zip(upserts, vectors, strict=True)
        ]
        graph.query(
            f"""
            UNWIND $rows AS row
            MERGE (c:{label} {{source: row.source, chunk_idx: row.chunk_idx}})
            SET c.text = row.text,
                c.content_hash = row.content_hash,
                c.embedding = row.embedding,
                c.title = row.title,
                c.department = row.department,
                c.acl_groups = row.acl_groups,
                c.effective_date = row.effective_date,
                c.superseded_by = row.superseded_by
            """,
            {"rows": payload},
        )

    if orphans:
        # DETACH: these chunks carry MENTIONS edges into the GraphRAG entity
        # graph. Deleting the node without its relationships would fail, and
        # leaving them would keep a withdrawn policy reachable through entities.
        graph.query(
            """
            UNWIND $ids AS element_id
            MATCH (c) WHERE elementId(c) = element_id
            DETACH DELETE c
            """,
            {"ids": orphans},
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the policy corpus into Neo4j.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete every node first. Rarely needed now that ingest is incremental, "
        "and it also destroys the GraphRAG entity graph built on these nodes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing or embedding.",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    graph = connect()
    if args.reset and not args.dry_run:
        existing = graph.query("MATCH (n) RETURN count(n) AS count")[0]["count"]
        if existing:
            graph.query("MATCH (n) DETACH DELETE n")
            print(f"Cleared {existing} existing nodes.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    totals: dict[str, int] = {}
    for department, docs in DOCUMENTS.items():
        chunks = build_chunks(docs, splitter)
        stats = sync_department(graph, department, chunks, dry_run=args.dry_run)
        name = INDEX_CONFIG[department]["department_name"]
        print(
            f"{name}: {len(docs)} documents -> {stats['total']} chunks "
            f"(new {stats['new']}, changed {stats['changed']}, "
            f"unchanged {stats['unchanged']}, deleted {stats['deleted']})"
        )
        for metric, value in stats.items():
            totals[metric] = totals.get(metric, 0) + value

    wrote = totals["new"] + totals["changed"] + totals["deleted"]
    print(f"{'would write' if args.dry_run else 'wrote'} {wrote} change(s)")
    if wrote == 0:
        print("corpus and graph are in sync; nothing to do")


if __name__ == "__main__":
    main()
