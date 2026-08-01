"""GraphRAG query modes over the community graph built by `scripts/graph_index.py`.

Two retrieval modes, following the Microsoft GraphRAG design:

- **global search** answers questions whose evidence is spread across many
  documents ("what requires VP approval?"). It map-reduces over community
  summaries rather than over chunks, so the answer is not limited by how many
  chunks fit in a top-k window.
- **local search** answers questions anchored on specific entities. It starts
  from the entities named in the question and walks out to their neighbours and
  source chunks.

Global search is the one that matters here: the failure the eval measures is
that top-k=4 vector retrieval cannot assemble an answer that lives in nine
documents, and no amount of reranking fixes a k that is too small.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict, cast

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel, Field

from app.config import settings
from app.observability import langchain_config
from app.retrieval.rag import get_embedder

MAP_SYSTEM = """You are answering part of a question using one summary of a group of company policies.

Extract only what this summary actually supports. Do not guess or generalise.
If the summary says nothing relevant, return an empty points list and score 0.
Each point should be a specific, self-contained fact — name the threshold,
approver, deadline, or system involved."""

REDUCE_SYSTEM = """You are answering a company policy question by combining findings from several groups of policies.

Rules:
- Use only the supplied findings. Never invent a threshold, approver, or deadline.
- The question usually spans several policies; give a complete consolidated answer
  rather than stopping at the first item.
- Group related items and state the specific dollar amounts, deadlines, and
  approver roles verbatim.
- If the findings genuinely do not answer the question, say you don't have that
  information in your policies."""


class MapPoint(BaseModel):
    point: str = Field(description="A specific fact from this summary that helps answer the question.")
    score: int = Field(description="0-100 how directly this helps answer the question.")


class MapResult(BaseModel):
    points: list[MapPoint]


class GraphAnswer(TypedDict):
    answer: str
    sources: list[str]
    communities: list[str]
    contexts: list[str]
    mode: str


#: The map step needs structured output; the configured chat model cannot do it.
#: `nemotron-3-super-120b` is a reasoning model and its thinking trace makes
#: LangChain return None from `with_structured_output` — the same failure that
#: broke ragas scoring and graph extraction. The reduce step writes prose, so it
#: keeps using the configured chat model.
MAP_MODEL = "meta/llama-3.1-8b-instruct"

#: The reduce step writes the final prose. The configured chat model leaks raw
#: reasoning scaffold ("We need to answer:" followed by a run of <unk> tokens)
#: on the long findings prompt. mistral-nemotron is non-reasoning and fine at
#: plain generation — it only fails at tool calling, which reduce never uses.
REDUCE_MODEL = "mistralai/mistral-nemotron"


def get_map_llm() -> Any:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA(
        model=MAP_MODEL,
        api_key=settings.nvidia_api_key.get_secret_value(),
        temperature=0.0,
    )


def get_reduce_llm() -> Any:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA(
        model=REDUCE_MODEL,
        api_key=settings.nvidia_api_key.get_secret_value(),
        temperature=0.0,
    )


def connect() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )


def rank_communities(
    graph: Neo4jGraph, question: str, top_k: int, level: int
) -> list[dict[str, Any]]:
    """Rank community summaries against the question by cosine similarity."""
    rows = graph.query(
        """
        MATCH (c:Community)
        WHERE c.embedding IS NOT NULL AND c.level >= $level
        RETURN c.id AS id, c.title AS title, c.summary AS summary, c.embedding AS embedding
        """,
        {"level": level},
    )
    if not rows:
        return []
    matrix = np.array([row["embedding"] for row in rows], dtype=float)
    query_vector = np.array(get_embedder().embed_query(question), dtype=float)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query_vector /= np.linalg.norm(query_vector)
    scores = matrix @ query_vector
    order = np.argsort(-scores)[:top_k]
    return [rows[i] for i in order]


def community_sources(graph: Neo4jGraph, community_ids: list[str], min_entities: int = 3) -> list[str]:
    """Documents substantially covered by the contributing communities.

    Requires a document to mention several of a community's entities, not just
    one. Hub entities ("Manager", "Finance", "Employee") are mentioned by nearly
    every policy, so a single shared entity attributes the entire corpus — the
    first version of this returned 70 of 75 documents and would have scored a
    perfect, meaningless source recall.
    """
    rows = graph.query(
        """
        UNWIND $ids AS cid
        MATCH (:Community {id: cid})<-[:IN_COMMUNITY]-(e:Entity)<-[:MENTIONS]-(chunk)
        WITH chunk.source AS source, count(DISTINCT e) AS hits
        WHERE source IS NOT NULL AND hits >= $min_entities
        RETURN source ORDER BY hits DESC
        """,
        {"ids": community_ids, "min_entities": min_entities},
    )
    return [row["source"] for row in rows]


def global_search(
    question: str,
    top_communities: int = 8,
    level: int = 1,
    workers: int = 4,
) -> GraphAnswer:
    """Map-reduce over the most relevant community summaries."""
    graph = connect()
    communities = rank_communities(graph, question, top_communities, level)
    if not communities:
        return {
            "answer": "I don't have that information in my policies.",
            "sources": [],
            "communities": [],
            "contexts": [],
            "mode": "global",
        }

    map_chain = (
        ChatPromptTemplate.from_messages(
            [("system", MAP_SYSTEM), ("human", "Question: {question}\n\nPolicy summary:\n{summary}")]
        )
        | get_map_llm().with_structured_output(MapResult)
    )

    def run_map(community: dict[str, Any]) -> tuple[str, list[MapPoint]]:
        try:
            result = cast(
                MapResult | None,
                map_chain.invoke(
                    {
                        "question": question,
                        "summary": f"{community['title']}. {community['summary']}",
                    },
                    config=langchain_config("graphrag_map"),
                ),
            )
        except Exception:  # noqa: BLE001 - one dead community must not kill the query
            return community["id"], []
        # Structured output returns None when the model emits an unparseable
        # tool call; that is a dropped community, not a failed query.
        return community["id"], result.points if result else []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        mapped = list(pool.map(run_map, communities))

    contributing = [cid for cid, group in mapped if any(p.score > 0 for p in group)]
    points = [point for _, group in mapped for point in group if point.score > 0]
    points.sort(key=lambda p: p.score, reverse=True)
    if not points:
        return {
            "answer": "I don't have that information in my policies.",
            "sources": [],
            "communities": [c["id"] for c in communities],
            "contexts": [],
            "mode": "global",
        }

    findings = "\n".join(f"- {point.point}" for point in points[:40])
    reduce_chain = (
        ChatPromptTemplate.from_messages(
            [("system", REDUCE_SYSTEM), ("human", "Question: {question}\n\nFindings:\n{findings}")]
        )
        | get_reduce_llm()
    )
    answer = reduce_chain.invoke(
        {"question": question, "findings": findings},
        config=langchain_config("graphrag_reduce"),
    )
    return {
        "answer": str(answer.content),
        "sources": community_sources(graph, contributing),
        "communities": contributing,
        # The findings the reduce step actually saw. Faithfulness must be judged
        # against what the model was given, not against the whole graph.
        "contexts": [point.point for point in points[:40]],
        "mode": "global",
    }


def local_search(question: str, top_entities: int = 32) -> GraphAnswer:
    """Answer from the neighbourhood of the entities the question names.

    The entity budget is wide on purpose. An inventory question ("what requires
    VP approval?") has its answer spread over 19 documents, so a narrow top-k
    reproduces exactly the completeness failure that vector retrieval already
    has — at 12 entities the answers were correct but missed items.

    Entities are matched with Neo4j's fulltext index over entity names rather
    than by embedding: the question usually contains the entity almost verbatim
    ("VP approval", "Coupa", "$25,000"), and exact-ish matching beats a nearest
    neighbour that returns a plausible but different threshold.
    """
    graph = connect()
    rows = graph.query(
        """
        CALL db.index.fulltext.queryNodes('entity_names', $query) YIELD node, score
        WITH node, score ORDER BY score DESC LIMIT $limit
        OPTIONAL MATCH (node)<-[:MENTIONS]-(chunk)
        OPTIONAL MATCH (node)-[r:RELATED]-(neighbour:Entity)
        RETURN node.name AS name, node.type AS type, node.description AS description,
               collect(DISTINCT chunk.source) AS sources,
               collect(DISTINCT chunk.text)[..2] AS texts,
               collect(DISTINCT neighbour.name + ': ' + coalesce(r.description, ''))[..6] AS neighbours
        """,
        {"query": _fulltext_query(question), "limit": top_entities},
    )
    if not rows:
        return {
            "answer": "I don't have that information in my policies.",
            "sources": [],
            "communities": [],
            "contexts": [],
            "mode": "local",
        }

    context_parts = []
    sources: set[str] = set()
    for row in rows:
        sources.update(source for source in row["sources"] if source)
        context_parts.append(
            f"{row['name']} ({row['type']}): {row['description']}\n"
            f"  related: {'; '.join(row['neighbours'])}\n"
            f"  policy text: {' '.join(row['texts'])}"
        )

    chain = (
        ChatPromptTemplate.from_messages(
            [("system", REDUCE_SYSTEM), ("human", "Question: {question}\n\nContext:\n{context}")]
        )
        | get_reduce_llm()
    )
    answer = chain.invoke(
        {"question": question, "context": "\n\n".join(context_parts)},
        config=langchain_config("graphrag_local"),
    )
    return {
        "answer": str(answer.content),
        "sources": sorted(sources),
        "communities": [],
        "contexts": context_parts,
        "mode": "local",
    }


def _fulltext_query(question: str) -> str:
    """Turn a question into a safe Lucene OR query.

    Lucene treats ?, :, $, and friends as operators, so a raw question is both a
    syntax error and a poor query.
    """
    words = [
        "".join(character for character in word if character.isalnum())
        for word in question.split()
    ]
    return " OR ".join(word for word in words if len(word) > 2)
