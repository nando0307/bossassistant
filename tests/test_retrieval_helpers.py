from __future__ import annotations

from langchain_core.documents import Document

from app.config import settings
from app.retrieval.rag import dedupe_sources, format_sources, model_for_mode


def test_model_for_mode_uses_fast_and_deep_settings() -> None:
    assert model_for_mode("fast") == settings.nvidia_chat_model
    assert model_for_mode("deep") == settings.nvidia_deep_chat_model


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
