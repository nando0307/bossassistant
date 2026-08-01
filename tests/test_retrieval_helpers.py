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


# ── ACL filtering ────────────────────────────────────────────────────


def _doc(source: str, groups: list[str] | None) -> Document:
    meta: dict[str, object] = {"source": source}
    if groups is not None:
        meta["acl_groups"] = groups
    return Document(page_content=f"body of {source}", metadata=meta)


def test_filter_by_acl_keeps_only_readable_chunks() -> None:
    from app.retrieval.rag import filter_by_acl

    docs = [
        _doc("HR-001", ["all-employees"]),
        _doc("FIN-037", ["executives"]),
        _doc("HR-011", ["managers", "hr-team"]),
    ]
    kept = {d.metadata["source"] for d in filter_by_acl(docs, frozenset({"all-employees"}))}
    assert kept == {"HR-001"}

    kept = {d.metadata["source"] for d in filter_by_acl(docs, frozenset({"all-employees", "executives"}))}
    assert kept == {"HR-001", "FIN-037"}


def test_filter_by_acl_fails_closed_on_unstamped_chunks() -> None:
    """A chunk with no ACL is a bug upstream; treat it as unreadable.

    Ingestion stamps every document, so defaulting an unstamped chunk to public
    would turn an ingestion regression into a silent disclosure.
    """
    from app.retrieval.rag import filter_by_acl

    assert filter_by_acl([_doc("HR-001", None)], frozenset({"all-employees"})) == []
    assert filter_by_acl([_doc("HR-001", [])], frozenset({"all-employees"})) == []


def test_filter_by_acl_disabled_when_groups_is_none() -> None:
    """None means "no enforcement" for offline tooling — distinct from empty."""
    from app.retrieval.rag import filter_by_acl

    docs = [_doc("FIN-037", ["executives"])]
    assert filter_by_acl(docs, None) == docs
    assert filter_by_acl(docs, frozenset()) == []


def test_every_corpus_document_is_acl_stamped() -> None:
    """An unstamped document becomes an unreachable chunk after ingest."""
    from app.corpus import FIN_DOCS, HR_DOCS

    for doc in (*HR_DOCS, *FIN_DOCS):
        groups = doc.metadata.get("acl_groups")
        assert groups, f"{doc.metadata['source']} has no acl_groups"


def test_restricted_documents_are_not_readable_by_all_employees() -> None:
    """The ACL axis must not collapse onto department.

    If every Finance document were readable by every Finance user, the existing
    per-department indexes would already be the whole feature.
    """
    from app.corpus import FIN_DOCS, HR_DOCS

    by_source = {d.metadata["source"]: d.metadata["acl_groups"] for d in (*HR_DOCS, *FIN_DOCS)}
    assert "all-employees" not in by_source["FIN-037"]  # M&A
    assert "all-employees" not in by_source["HR-011"]  # compensation bands
    assert "all-employees" in by_source["HR-001"]  # PTO
