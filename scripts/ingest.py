"""Load the HR and Finance policy corpus into Neo4j.

The corpus itself lives in `app.corpus` so the vector ingester and the
GraphRAG indexer cannot drift apart.

Index names, node labels, and the embedder come from the retrieval module so
ingestion and query time cannot drift apart.
"""
from __future__ import annotations

import argparse

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.corpus import DOCUMENTS
from app.retrieval.rag import INDEX_CONFIG, get_embedder

def connect() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the policy corpus into Neo4j.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete every node in the graph before ingesting. Required to re-ingest.",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    graph = connect()
    existing = graph.query("MATCH (n) RETURN count(n) AS count")[0]["count"]
    if existing and not args.reset:
        # Re-running without a wipe duplicates every chunk, which silently
        # skews retrieval instead of failing.
        raise SystemExit(f"graph already holds {existing} nodes; re-run with --reset to replace them")
    if args.reset and existing:
        graph.query("MATCH (n) DETACH DELETE n")
        print(f"Cleared {existing} existing nodes.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    for department, docs in DOCUMENTS.items():
        config = INDEX_CONFIG[department]
        chunks = splitter.split_documents(docs)
        Neo4jVector.from_documents(
            documents=chunks,
            embedding=get_embedder(),
            index_name=config["index_name"],
            keyword_index_name=config["keyword_index_name"],
            node_label=config["node_label"],
            text_node_property="text",
            embedding_node_property="embedding",
            search_type="hybrid",
            url=settings.neo4j_uri,
            username=settings.neo4j_user,
            password=settings.neo4j_password.get_secret_value(),
            database=settings.neo4j_database,
        )
        print(f"{config['department_name']}: {len(docs)} documents -> {len(chunks)} chunks")

    print(graph.query("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count"))


if __name__ == "__main__":
    main()
