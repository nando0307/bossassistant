from __future__ import annotations

from langchain_core.documents import Document

from app.retrieval.rag import dedupe_sources, format_sources


def test_format_sources_dedupes_duplicate_document_chunks() -> None:
    docs = [
        Document(
            page_content="\ntext: First PTO chunk.",
            metadata={"source": "HR-001", "title": "PTO Policy", "department": "hr"},
        ),
        Document(
            page_content="\ntext: Second PTO chunk.",
            metadata={"source": "HR-001", "title": "PTO Policy", "department": "hr"},
        ),
        Document(
            page_content="\ntext: Travel policy chunk.",
            metadata={"source": "FIN-002", "title": "Corporate Travel Policy", "department": "finance"},
        ),
    ]

    sources = format_sources(docs)

    assert [source["source"] for source in sources] == ["HR-001", "FIN-002"]
    assert sources[0]["preview"] == "First PTO chunk."


def test_dedupe_sources_preserves_entries_without_source_id() -> None:
    sources = [
        {"source": None, "title": "Unknown", "department": None, "preview": "first"},
        {"source": None, "title": "Unknown", "department": None, "preview": "second"},
    ]

    assert dedupe_sources(sources) == sources
