"""Build the GraphRAG knowledge graph over the policy corpus.

Pipeline, following the Microsoft GraphRAG design:

    chunks -> entity/relationship extraction -> entity resolution
           -> Leiden communities (hierarchical) -> community summaries

Clustering runs client-side in networkx. Neo4j Aura lists the GDS procedures,
but calling `gds.graph.project` fails asking for a `sessionId`: GDS on Aura is
the separately provisioned Graph Analytics Serverless product, not an
in-database library. networkx is already an installed transitive dependency, so
the `graphrag` package, `graspologic`, and `leidenalg` all stay unnecessary.

Chunks are the nodes the vector ingester already wrote (`HRDocument` /
`FinDocument`), so text and embeddings are not duplicated — this graph layers
on top of `scripts/ingest.py` rather than replacing it.

Extraction is checkpointed to `--cache`, keyed by a hash of the chunk text rather
than its Neo4j elementId: elementIds are reassigned by `ingest.py --reset`, which
would silently invalidate the whole cache exactly when re-ingesting. The NVIDIA endpoint throttles hard
(`ResourceExhausted: Worker local total request limit reached (107/32)`), and
re-running 142 extraction calls because call 130 got a 503 is how an index
build becomes a two-hour job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel, Field

from app.config import settings

EXTRACTION_SYSTEM = """You extract a knowledge graph from corporate policy documents.

Identify every significant entity and the relationships between them.

Entity types to use:
- ROLE: an approver or actor (Manager, VP, CFO, Board, HR, Legal, Finance, Internal Audit)
- SYSTEM: a named tool or platform (Workday, Expensify, Coupa, Navan, EthicsPoint)
- POLICY_TOPIC: the subject area (Travel, Procurement, Parental Leave, SOX Compliance)
- THRESHOLD: a monetary or numeric limit ($25,000, $500/night, 110% of budget)
- DEADLINE: a time constraint (30 days, November 30, 5 business days, Net 30)
- OBLIGATION: something required or prohibited (VP approval, dual approval, security review)

Rules:
- Use the exact name as written in the text. Do not invent entities.
- Write descriptions as a single factual sentence grounded in the text.
- Relationships must connect two entities you also returned.
- Prefer relationships that cross topics: which ROLE approves which THRESHOLD,
  which OBLIGATION applies to which POLICY_TOPIC, which SYSTEM a process runs in.
- strength is 1-10 for how central the relationship is to the document."""


class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name exactly as written in the text.")
    type: str = Field(description="One of ROLE, SYSTEM, POLICY_TOPIC, THRESHOLD, DEADLINE, OBLIGATION.")
    description: str = Field(description="One factual sentence grounded in the text.")


class ExtractedRelationship(BaseModel):
    source: str = Field(description="Name of the source entity.")
    target: str = Field(description="Name of the target entity.")
    description: str = Field(description="How the two are related, grounded in the text.")
    strength: int = Field(description="1-10 importance of this relationship.")


class ChunkGraph(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


class CommunitySummary(BaseModel):
    title: str = Field(description="Short name for what this group of policies covers.")
    summary: str = Field(description="A paragraph explaining the theme and its key rules, thresholds, and approvers.")
    rating: int = Field(description="1-10 how important this theme is for someone needing to follow company policy.")


def connect() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )


def get_extractor(model: str) -> Any:
    """Build the extraction chain.

    Imported lazily so `--help` and the Cypher-only stages do not pay for the
    NVIDIA client import.
    """
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    llm = ChatNVIDIA(
        model=model,
        api_key=settings.nvidia_api_key.get_secret_value(),
        temperature=0.0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", EXTRACTION_SYSTEM), ("human", "Document {source}:\n\n{text}")]
    )
    return prompt | llm.with_structured_output(ChunkGraph)


def get_summarizer(model: str) -> Any:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    llm = ChatNVIDIA(
        model=model,
        api_key=settings.nvidia_api_key.get_secret_value(),
        temperature=0.0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You summarize a cluster of related corporate policy entities into a "
                "standalone report. Name the concrete thresholds, approvers, deadlines, "
                "and systems involved. Someone must be able to answer policy questions "
                "from your summary alone, without the source documents.",
            ),
            ("human", "Entities and relationships in this cluster:\n\n{content}"),
        ]
    )
    return prompt | llm.with_structured_output(CommunitySummary)


def call_with_retry(chain: Any, payload: dict[str, Any], attempts: int = 5) -> Any:
    """Retry with exponential backoff, returning None if every attempt fails.

    Two failure modes, both retried the same way: the endpoint returns 503
    ResourceExhausted under load rather than a proper 429, and the 8B model
    intermittently emits an unparseable tool call, which LangChain surfaces as
    a None result rather than an exception. A None that reaches the caller
    would crash the whole build over one bad chunk.
    """
    delay = 2.0
    last: str = "no attempts made"
    for _ in range(attempts):
        try:
            result = chain.invoke(payload)
            if result is not None:
                return result
            last = "structured output returned None"
        except Exception as exc:  # noqa: BLE001 - provider raises bare Exception
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(delay)
        delay *= 2
    print(f"  WARNING: giving up on a chunk after {attempts} attempts ({last[:120]})")
    return None


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cached = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            cached[record["chunk_id"]] = record["graph"]
    return cached


def extract_all(
    graph: Neo4jGraph, model: str, cache_path: Path, workers: int
) -> dict[str, dict[str, Any]]:
    chunks = graph.query(
        """
        MATCH (c)
        WHERE c:HRDocument OR c:FinDocument
        RETURN elementId(c) AS chunk_id, c.text AS text, c.source AS source
        """
    )
    for chunk in chunks:
        chunk["key"] = hashlib.sha256(chunk["text"].encode()).hexdigest()[:16]
    cache = load_cache(cache_path)
    todo = [c for c in chunks if c["key"] not in cache]
    print(f"chunks={len(chunks)} cached={len(cache)} to_extract={len(todo)}")
    if not todo:
        return cache

    extractor = get_extractor(model)
    handle = cache_path.open("a")

    def run(chunk: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        result: ChunkGraph | None = call_with_retry(
            extractor, {"text": chunk["text"], "source": chunk["source"]}
        )
        return chunk["key"], result.model_dump() if result else None

    done = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk_id, extracted in pool.map(run, todo):
            done += 1
            if extracted is None:
                skipped += 1
                continue
            cache[chunk_id] = extracted
            handle.write(json.dumps({"chunk_id": chunk_id, "graph": extracted}) + "\n")
            handle.flush()
            if done % 20 == 0:
                print(f"  extracted {done}/{len(todo)} (skipped {skipped})")
    handle.close()
    if skipped:
        print(f"WARNING: {skipped}/{len(todo)} chunks failed extraction; re-run to retry them")
    return cache


def write_graph(graph: Neo4jGraph, extracted: dict[str, dict[str, Any]]) -> None:
    """Write entities and relationships, merging on the normalized name.

    Normalizing to upper case is the entity-resolution step. It is deliberately
    crude: "VP", "VP-level", and "Vice President" stay distinct. Embedding-based
    merging is the upgrade path if the graph turns out to be fragmented.
    """
    graph.query("MATCH (e:Entity) DETACH DELETE e")
    graph.query("MATCH (c:Community) DETACH DELETE c")

    chunks = graph.query(
        "MATCH (c) WHERE c:HRDocument OR c:FinDocument "
        "RETURN elementId(c) AS chunk_id, c.text AS text"
    )
    rows = []
    for chunk in chunks:
        key = hashlib.sha256(chunk["text"].encode()).hexdigest()[:16]
        data = extracted.get(key)
        if data:
            rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "entities": data["entities"],
                    "relationships": data["relationships"],
                }
            )
    graph.query(
        """
        UNWIND $rows AS row
        MATCH (c) WHERE elementId(c) = row.chunk_id
        UNWIND row.entities AS entity
        MERGE (e:Entity {key: toUpper(trim(entity.name))})
          ON CREATE SET e.name = entity.name, e.type = entity.type, e.description = entity.description
        MERGE (c)-[:MENTIONS]->(e)
        """,
        {"rows": rows},
    )
    graph.query(
        """
        UNWIND $rows AS row
        UNWIND row.relationships AS rel
        MATCH (a:Entity {key: toUpper(trim(rel.source))})
        MATCH (b:Entity {key: toUpper(trim(rel.target))})
        MERGE (a)-[r:RELATED]->(b)
          ON CREATE SET r.description = rel.description, r.weight = toFloat(rel.strength)
          ON MATCH SET r.weight = r.weight + toFloat(rel.strength)
        """,
        {"rows": rows},
    )
    stats = graph.query(
        "MATCH (e:Entity) WITH count(e) AS entities "
        "MATCH ()-[r:RELATED]->() RETURN entities, count(r) AS relationships"
    )
    print(f"graph written: {stats}")


def detect_communities(graph: Neo4jGraph) -> None:
    """Hierarchical Leiden over the entity graph.

    Runs client-side in networkx rather than through Neo4j GDS. The GDS
    procedures are listed on this Aura instance, but calling `gds.graph.project`
    fails asking for a `sessionId` — GDS on Aura is the separately provisioned
    Graph Analytics Serverless product, not an in-database library. networkx
    3.6 ships `leiden_partitions`, which yields the dendrogram levels this
    pipeline needs, and networkx is already an installed transitive dependency.
    At 596 entities the client-side round trip is instant.
    """
    import networkx as nx  # type: ignore[import-untyped]

    edges = graph.query(
        """
        MATCH (a:Entity)-[r:RELATED]->(b:Entity)
        RETURN a.key AS source, b.key AS target, coalesce(r.weight, 1.0) AS weight
        """
    )
    nodes = graph.query("MATCH (e:Entity) RETURN e.key AS key")
    network = nx.Graph()
    network.add_nodes_from(row["key"] for row in nodes)
    for row in edges:
        network.add_edge(row["source"], row["target"], weight=float(row["weight"]))

    # Each successive partition is one level coarser; level 0 is the most
    # granular. Isolated entities form singleton communities and are dropped
    # later by --min-community-size.
    levels = list(nx.community.louvain_partitions(network, weight="weight", seed=7))
    print(f"louvain: {len(levels)} levels, sizes={[len(partition) for partition in levels]}")

    assignments = []
    for level, partition in enumerate(levels):
        for community_index, members in enumerate(partition):
            for key in members:
                assignments.append(
                    {"key": key, "level": level, "cid": f"{level}-{community_index}"}
                )
    graph.query(
        """
        UNWIND $rows AS row
        MATCH (e:Entity {key: row.key})
        MERGE (c:Community {id: row.cid})
          ON CREATE SET c.level = row.level
        MERGE (e)-[:IN_COMMUNITY]->(c)
        """,
        {"rows": assignments},
    )

    # Parent links let a summary at level N cite the level N-1 groups beneath it.
    graph.query(
        """
        MATCH (e:Entity)-[:IN_COMMUNITY]->(child:Community)
        MATCH (e)-[:IN_COMMUNITY]->(parent:Community)
        WHERE parent.level = child.level + 1
        MERGE (child)-[:PARENT_OF]->(parent)
        """
    )
    print(graph.query(
        "MATCH (c:Community) RETURN c.level AS level, count(*) AS communities ORDER BY level"
    ))


def summarize_communities(
    graph: Neo4jGraph, model: str, workers: int, min_size: int, min_level: int
) -> None:
    communities = graph.query(
        """
        MATCH (c:Community)<-[:IN_COMMUNITY]-(e:Entity)
        WITH c, collect(e) AS members
        WHERE size(members) >= $min_size AND c.level >= $min_level
        RETURN c.id AS id, c.level AS level,
               [m IN members | m.name + ' (' + m.type + '): ' + coalesce(m.description, '')] AS entities,
               [(a)-[r:RELATED]->(b) WHERE a IN members AND b IN members |
                 a.name + ' -> ' + b.name + ': ' + coalesce(r.description, '')] AS relationships
        """,
        {"min_size": min_size, "min_level": min_level},
    )
    print(f"summarizing {len(communities)} communities (min_size={min_size}, min_level={min_level})")
    if not communities:
        return

    summarizer = get_summarizer(model)

    def run(community: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        content = "Entities:\n" + "\n".join(community["entities"][:60])
        if community["relationships"]:
            content += "\n\nRelationships:\n" + "\n".join(community["relationships"][:60])
        result: CommunitySummary | None = call_with_retry(summarizer, {"content": content})
        return community["id"], result.model_dump() if result else None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for community_id, summary in pool.map(run, communities):
            if summary is None:
                continue
            graph.query(
                """
                MATCH (c:Community {id: $id})
                SET c.title = $title, c.summary = $summary, c.rating = $rating
                """,
                {"id": community_id, **summary},
            )
            done += 1
            if done % 10 == 0:
                print(f"  summarized {done}/{len(communities)}")
    print(f"summarized {done} communities")


def embed_summaries(graph: Neo4jGraph) -> None:
    """Store an embedding per community summary.

    Global search otherwise has to map over every community for every question.
    Ranking summaries against the question first, and only mapping over the top
    few, is what keeps a global query to a handful of LLM calls instead of 55.
    With this few communities a stored vector index is overkill — the scorer
    fetches all of them and ranks in numpy.
    """
    from app.retrieval.rag import get_embedder

    rows = graph.query(
        "MATCH (c:Community) WHERE c.summary IS NOT NULL "
        "RETURN c.id AS id, c.title AS title, c.summary AS summary"
    )
    if not rows:
        print("no summaries to embed")
        return
    embedder = get_embedder()
    vectors = embedder.embed_documents([f"{r['title']}. {r['summary']}" for r in rows])
    graph.query(
        """
        UNWIND $rows AS row
        MATCH (c:Community {id: row.id})
        SET c.embedding = row.embedding
        """,
        {"rows": [{"id": r["id"], "embedding": v} for r, v in zip(rows, vectors, strict=True)]},
    )
    print(f"embedded {len(rows)} community summaries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GraphRAG index.")
    parser.add_argument(
        "--model",
        default="meta/llama-3.1-8b-instruct",
        help="Chosen by measurement, not by size. The configured chat model "
        "(nemotron-3-super-120b) returns None from structured output because its "
        "reasoning trace breaks tool calling; mistral-nemotron 500s on tool calls "
        "entirely; llama-3.3-70b and nemotron-super-49b hit 503 ResourceExhausted "
        "or time out. llama-3.1-8b returns 16 entities and 13 relationships per "
        "chunk in ~8s and survives 142 sequential calls.",
    )
    parser.add_argument("--cache", type=Path, default=Path("evals/.graph_extract_cache.jsonl"))
    parser.add_argument("--workers", type=int, default=3, help="Kept low; the endpoint throttles at ~32 in-flight.")
    parser.add_argument("--min-community-size", type=int, default=3)
    parser.add_argument(
        "--min-community-level",
        type=int,
        default=1,
        help="Skip level 0: it is mostly 2-3 entity fragments, and global search "
        "reads the coarser levels where a summary spans several policies.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "extract", "write", "communities", "summarize", "embed"],
        default="all",
    )
    args = parser.parse_args()

    graph = connect()
    extracted: dict[str, dict[str, Any]] = {}

    if args.stage in {"all", "extract", "write"}:
        extracted = extract_all(graph, args.model, args.cache, args.workers)
    if args.stage in {"all", "write"}:
        write_graph(graph, extracted)
    if args.stage in {"all", "communities"}:
        detect_communities(graph)
    if args.stage in {"all", "summarize"}:
        summarize_communities(
            graph, args.model, args.workers, args.min_community_size, args.min_community_level
        )
    if args.stage in {"all", "summarize", "embed"}:
        embed_summaries(graph)


if __name__ == "__main__":
    main()
