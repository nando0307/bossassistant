"""Load the HR and Finance policy corpus into Neo4j, incrementally.

The corpus itself lives in `app.corpus` so the vector ingester and the GraphRAG
indexer cannot drift apart. Index names, node labels, and the embedder come from
the retrieval module so ingestion and query time cannot drift apart either.

**Why this is not `Neo4jVector.from_documents`.** That helper creates fresh
nodes on every call, which is fine exactly once. A policy corpus changes: a
document gets revised, another is withdrawn, a third is re-worded. Re-running a
create-only ingester either duplicates every chunk or forces a full wipe, and a
full wipe throws away the GraphRAG entity graph built on top of these nodes.

Chunks have stable `(source, chunk_idx)` identity and a text content hash:

* unchanged text and metadata -> no write or embedding
* metadata-only change -> update properties, preserve embedding
* changed/new text -> embed and upsert
* chunk no longer in the corpus -> delete

That last case is the one enterprises actually get bitten by. A withdrawn policy
whose chunks linger is not a stale cache, it is an assistant citing a rule that
no longer exists. Shortened documents hit the same path: if a document used to
split into 4 chunks and now splits into 2, chunks 2 and 3 are orphans.

Every write runs inside a shared, drained maintenance window. Publication follows
full corpus verification; failures leave admission closed until a verified retry.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
import logging
from typing import Any

from langchain_neo4j import Neo4jGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.corpus import DOCUMENTS
from app.policy import gate
from app.retrieval.rag import EMBEDDING_DIMENSIONS, INDEX_CONFIG, Department, get_embedder

#: Declared here because the vector index must be created with the right width
#: before any node is written. The width itself lives next to `get_embedder`,
#: which requests exactly this many dimensions from the model.


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
        refresh_schema=False,
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


def metadata_properties(chunk: Chunk) -> dict[str, Any]:
    return {
        "title": chunk.metadata.get("title"),
        "department": chunk.metadata.get("department"),
        "acl_groups": sorted(set(chunk.metadata.get("acl_groups") or [])),
        "effective_date": chunk.metadata.get("effective_date"),
        "superseded_by": chunk.metadata.get("superseded_by"),
    }


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
        "c.chunk_idx AS chunk_idx, coalesce(c.content_hash, '') AS content_hash, "
        "properties(c) AS properties"
    )


def sync_department(
    graph: Neo4jGraph,
    department: Department,
    chunks: list[Chunk],
    dry_run: bool = False,
    embedder: Any | None = None,
    *,
    maintenance: str | None = None,
) -> dict[str, int]:
    """Bring one department's chunks in line with the corpus. Returns a diff.

    `embedder` is injected so the diff logic can be tested without a live
    endpoint; production passes None and gets the configured embedder.
    """
    if not dry_run and (maintenance is None or not gate.drained(maintenance)):
        raise RuntimeError("Ingestion requires an owned, drained maintenance window")
    label = INDEX_CONFIG[department]["node_label"]
    if not dry_run:
        ensure_indexes(graph, department)

    rows = existing_chunks(graph, department)
    wanted = {chunk.key: chunk for chunk in chunks}
    if len(wanted) != len(chunks):
        raise ValueError("Duplicate source/chunk identity in corpus")
    before = {
        (row["source"], row["chunk_idx"]): row
        for row in rows
        if row["source"] is not None and row["chunk_idx"] is not None
    }
    if len(before) != sum(row["source"] is not None and row["chunk_idx"] is not None for row in rows):
        raise ValueError("Duplicate chunks in graph; repair while serving is closed")
    upserts = [c for c in chunks if before.get(c.key, {}).get("content_hash") != c.content_hash]
    metadata_updates = [
        c for c in chunks
        if c.key in before and before[c.key]["content_hash"] == c.content_hash
        and any(before[c.key].get("properties", {}).get(k) != v for k, v in metadata_properties(c).items())
    ]
    unchanged = len(chunks) - len(upserts) - len(metadata_updates)
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
        "metadata_changed": len(metadata_updates),
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
                **metadata_properties(c),
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

    if metadata_updates:
        graph.query(
            f"UNWIND $rows AS row "
            f"MATCH (c:{label} {{source: row.source, chunk_idx: row.chunk_idx}}) "
            "SET c += row.metadata",
            {"rows": [
                {"source": c.source, "chunk_idx": c.chunk_idx, "metadata": metadata_properties(c)}
                for c in metadata_updates
            ]},
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


def verify_corpus(graph: Neo4jGraph, corpus: dict[Department, list[Chunk]]) -> None:
    """Compare stored text, every served property and embeddings, not just hashes."""
    for department, chunks in corpus.items():
        rows = existing_chunks(graph, department)
        expected = {c.key: c for c in chunks}
        if len(rows) != len(expected):
            raise RuntimeError(f"{department}: corpus cardinality mismatch")
        seen = set()
        for row in rows:
            key = (row["source"], row["chunk_idx"])
            chunk = expected.get(key)
            if chunk is None or key in seen:
                raise RuntimeError(f"{department}: unexpected or duplicate chunk {key}")
            seen.add(key)
            props = row["properties"]
            if props.get("text") != chunk.text or props.get("content_hash") != chunk.content_hash:
                raise RuntimeError(f"{department}: text mismatch for {key}")
            if any(props.get(k) != v for k, v in metadata_properties(chunk).items()):
                raise RuntimeError(f"{department}: metadata mismatch for {key}")
            vector = props.get("embedding")
            if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSIONS:
                raise RuntimeError(f"{department}: missing/invalid embedding for {key}")
    graph.query("CALL db.awaitIndexes(300)")


def ingest_corpus(
    graph: Neo4jGraph,
    documents: dict[Department, list[Any]],
    splitter: RecursiveCharacterTextSplitter,
    *,
    embedder: Any | None = None,
    reset: bool = False,
    drain_timeout: float = 300,
    recover_stopped_workers: str | None = None,
) -> dict[str, int]:
    """The only publication path: close, drain, synchronize, verify, publish."""
    if set(documents) != set(INDEX_CONFIG):
        raise ValueError("Publication requires the complete HR and Finance corpus")
    corpus = {department: build_chunks(docs, splitter) for department, docs in documents.items()}
    owner = gate.close(recover_stopped_workers)
    print(f"Serving CLOSED; maintenance cycle: {owner}", flush=True)
    try:
        gate.drain(owner, drain_timeout)
        if reset:
            # Never erase the gate, its closed status or active registrations.
            graph.query("MATCH (n) WHERE NOT n:PolicyGate DETACH DELETE n")
        totals: dict[str, int] = {}
        for department, chunks in corpus.items():
            stats = sync_department(graph, department, chunks, embedder=embedder, maintenance=owner)
            for metric, value in stats.items():
                totals[metric] = totals.get(metric, 0) + value
        verify_corpus(graph, corpus)
        revision = gate.publish(owner)
        print(f"Verified corpus published at revision {revision}; serving OPEN", flush=True)
        return totals
    except BaseException:
        try:
            gate.fail(owner)
        except Exception:
            logging.exception("Maintenance state uncertain; do not resume serving without recovery")
        raise


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
    parser.add_argument("--drain-timeout", type=float, default=300)
    parser.add_argument(
        "--recover-stopped-workers",
        metavar="CLOSED_CYCLE",
        help="Recover abandoned registrations ONLY after stopping all old API and ingestion processes. "
        "Requires the exact closed maintenance cycle printed by the failed run.",
    )
    args = parser.parse_args()

    graph = connect()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    if args.dry_run:
        for department, docs in DOCUMENTS.items():
            print(department, sync_department(graph, department, build_chunks(docs, splitter), dry_run=True))
        return
    print(ingest_corpus(
        graph, DOCUMENTS, splitter, reset=args.reset, drain_timeout=args.drain_timeout,
        recover_stopped_workers=args.recover_stopped_workers,
    ))


if __name__ == "__main__":
    main()
