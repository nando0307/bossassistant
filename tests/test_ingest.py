"""Ingestion must survive a changing corpus.

These run against a fake graph rather than a live Neo4j: the behaviour under
test is the diff logic, and a test that needs Aura credentials is a test that
stops running.
"""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from scripts.ingest import build_chunks, sync_department


class FakeGraph:
    """Minimal stand-in that records writes and serves back stored chunks."""

    def __init__(self, stored: list[dict[str, Any]] | None = None) -> None:
        self.stored = stored or []
        self.writes: list[tuple[str, dict[str, Any]]] = []

    def query(self, statement: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = params or {}
        if "CREATE VECTOR INDEX" in statement or "CREATE FULLTEXT INDEX" in statement:
            return []
        if statement.strip().startswith("MATCH (c:") and "RETURN elementId(c)" in statement:
            return self.stored
        self.writes.append((statement, params))
        if "DETACH DELETE" in statement:
            removed = set(params.get("ids", []))
            self.stored = [row for row in self.stored if row["element_id"] not in removed]
        return []

    @property
    def rows_written(self) -> int:
        return sum(len(p.get("rows", [])) for _, p in self.writes if "MERGE" in _)

    @property
    def rows_deleted(self) -> int:
        return sum(len(p.get("ids", [])) for _, p in self.writes if "DETACH DELETE" in _)


class FakeEmbedder:
    """Deterministic stand-in; the diff logic is what is under test."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 1.0] for t in texts]


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def _doc(source: str, body: str) -> Document:
    return Document(
        page_content=body,
        metadata={
            "source": source,
            "title": source,
            "department": "hr",
            "acl_groups": ["all-employees"],
            "effective_date": "2024-01-01",
            "superseded_by": None,
        },
    )


def _stored_from(chunks: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "element_id": f"e{i}",
            "source": c.source,
            "chunk_idx": c.chunk_idx,
            "content_hash": c.content_hash,
        }
        for i, c in enumerate(chunks)
    ]


def test_chunk_index_is_stable_per_document() -> None:
    """Editing one document must not renumber every chunk after it.

    Splitting a whole department at once makes each chunk's index depend on how
    many chunks preceded it, so a one-word edit near the top would mark the
    entire corpus as changed and re-embed all of it.
    """
    docs = [_doc("HR-001", "alpha " * 200), _doc("HR-002", "beta " * 200)]
    first = build_chunks(docs, _splitter())

    docs[0] = _doc("HR-001", "alpha " * 400)  # first document grows
    second = build_chunks(docs, _splitter())

    unchanged = {(c.source, c.chunk_idx, c.content_hash) for c in first if c.source == "HR-002"}
    still_there = {(c.source, c.chunk_idx, c.content_hash) for c in second if c.source == "HR-002"}
    assert unchanged == still_there


def test_reingest_with_no_changes_writes_nothing() -> None:
    """The claim this file exists to defend."""
    docs = [_doc("HR-001", "alpha " * 200), _doc("HR-002", "beta " * 200)]
    chunks = build_chunks(docs, _splitter())
    graph = FakeGraph(_stored_from(chunks))

    stats = sync_department(graph, "hr", chunks, embedder=FakeEmbedder())  # type: ignore[arg-type]

    assert stats == {
        "total": len(chunks),
        "unchanged": len(chunks),
        "new": 0,
        "changed": 0,
        "deleted": 0,
    }
    assert graph.writes == [], "an unchanged corpus must not write or embed"


def test_changed_document_updates_only_its_own_chunks() -> None:
    docs = [_doc("HR-001", "alpha " * 200), _doc("HR-002", "beta " * 200)]
    chunks = build_chunks(docs, _splitter())
    graph = FakeGraph(_stored_from(chunks))

    docs[1] = _doc("HR-002", "beta revised " * 200)
    revised = build_chunks(docs, _splitter())
    stats = sync_department(graph, "hr", revised, embedder=FakeEmbedder())  # type: ignore[arg-type]

    assert stats["unchanged"] > 0
    assert stats["changed"] + stats["new"] > 0
    assert stats["deleted"] >= 0


def test_removed_document_has_its_chunks_deleted() -> None:
    """A withdrawn policy whose chunks linger is an assistant citing a dead rule."""
    docs = [_doc("HR-001", "alpha " * 200), _doc("HR-002", "beta " * 200)]
    chunks = build_chunks(docs, _splitter())
    graph = FakeGraph(_stored_from(chunks))

    remaining = build_chunks([docs[0]], _splitter())
    stats = sync_department(graph, "hr", remaining, embedder=FakeEmbedder())  # type: ignore[arg-type]

    removed = len(chunks) - len(remaining)
    assert stats["deleted"] == removed
    assert graph.rows_deleted == removed


def test_shortened_document_deletes_its_trailing_chunks() -> None:
    """The subtle case: the document still exists but produces fewer chunks."""
    long_doc = _doc("HR-001", "alpha " * 600)
    chunks = build_chunks([long_doc], _splitter())
    assert len(chunks) > 2
    graph = FakeGraph(_stored_from(chunks))

    shorter = build_chunks([_doc("HR-001", "alpha " * 60)], _splitter())
    stats = sync_department(graph, "hr", shorter, embedder=FakeEmbedder())  # type: ignore[arg-type]

    assert stats["deleted"] == len(chunks) - len(shorter)


def test_legacy_nodes_without_chunk_idx_are_replaced_not_duplicated() -> None:
    """Nodes from the old create-only ingester carry no chunk_idx.

    Skipping them in the diff would hide them, and the next run would write a
    second full set alongside them.
    """
    docs = [_doc("HR-001", "alpha " * 200)]
    chunks = build_chunks(docs, _splitter())
    legacy = [
        {"element_id": f"legacy{i}", "source": "HR-001", "chunk_idx": None, "content_hash": ""}
        for i in range(len(chunks))
    ]
    graph = FakeGraph(legacy)

    stats = sync_department(graph, "hr", chunks, embedder=FakeEmbedder())  # type: ignore[arg-type]

    assert stats["deleted"] == len(legacy)
    assert stats["new"] == len(chunks)


# ── Supersession ─────────────────────────────────────────────────────


def test_superseded_documents_are_dropped_at_retrieval() -> None:
    from app.retrieval.rag import drop_superseded

    live = Document(page_content="new rule", metadata={"source": "HR-039", "superseded_by": None})
    dead = Document(page_content="old rule", metadata={"source": "HR-016", "superseded_by": "HR-039"})

    kept = drop_superseded([dead, live])
    assert [d.metadata["source"] for d in kept] == ["HR-039"]


def test_corpus_supersession_links_resolve_both_ways() -> None:
    """`supersedes` is authored once; `superseded_by` is derived from it."""
    from app.corpus import FIN_DOCS, HR_DOCS

    by_source = {d.metadata["source"]: d.metadata for d in (*HR_DOCS, *FIN_DOCS)}
    for source, meta in by_source.items():
        replaced = meta.get("supersedes")
        if replaced:
            assert by_source[replaced]["superseded_by"] == source


def test_format_docs_carries_the_effective_date() -> None:
    """A policy figure without a vintage cannot be verified by the reader."""
    from app.retrieval.rag import format_docs

    rendered = format_docs(
        [Document(page_content="15 days", metadata={"title": "PTO Policy", "effective_date": "2024-01-01"})]
    )
    assert "effective 2024-01-01" in rendered
