"""Tests for document browsing over current and legacy Chroma records."""

from pathlib import Path

import pytest

from excort.ingestion import Chunk
from excort.vector_store import (
    DocumentCatalog,
    DocumentNotFoundError,
    add_document,
    rebuild_collection,
)


def make_chunk(
    chunk_id: str,
    *,
    source: str,
    page_number: int,
    chunk_index: int,
    document_id: str | None,
) -> Chunk:
    """Build a small chunk with internally consistent token metadata."""
    return Chunk(
        id=chunk_id,
        text=f"Text for {chunk_id}",
        source=source,
        page_number=page_number,
        chunk_index=chunk_index,
        token_start=chunk_index * 10,
        token_end=chunk_index * 10 + 10,
        token_count=10,
        document_id=document_id,
    )


def make_catalog(tmp_path: Path) -> DocumentCatalog:
    """Create a collection containing one legacy and one current document."""
    persist_path = tmp_path / "chroma"
    legacy_chunks = [
        make_chunk(
            "legacy-p0002-c0000",
            source="data/raw/legacy.pdf",
            page_number=2,
            chunk_index=0,
            document_id=None,
        ),
        make_chunk(
            "legacy-p0001-c0000",
            source="data/raw/legacy.pdf",
            page_number=1,
            chunk_index=0,
            document_id=None,
        ),
    ]
    rebuild_collection(
        persist_path,
        "test_documents",
        legacy_chunks,
        [[1.0, 0.0], [0.9, 0.1]],
    )
    current_chunks = [
        make_chunk(
            "abc-p0001-c0001",
            source="current.pdf",
            page_number=1,
            chunk_index=1,
            document_id="abc",
        ),
        make_chunk(
            "abc-p0001-c0000",
            source="current.pdf",
            page_number=1,
            chunk_index=0,
            document_id="abc",
        ),
    ]
    add_document(
        persist_path,
        "test_documents",
        current_chunks,
        [[0.0, 1.0], [0.1, 0.9]],
    )
    return DocumentCatalog(persist_path, "test_documents")


def test_catalog_lists_current_and_legacy_documents(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)

    documents = catalog.list_documents()

    assert [document.filename for document in documents] == [
        "current.pdf",
        "legacy.pdf",
    ]
    assert documents[0].document_id == "abc"
    assert documents[0].indexed_page_count == 1
    assert documents[0].chunk_count == 2
    assert documents[1].document_id.startswith("legacy-")
    assert documents[1].indexed_page_count == 2


def test_catalog_paginates_chunks_in_document_order(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)

    first = catalog.get_document("abc", page=1, page_size=1)
    second = catalog.get_document("abc", page=2, page_size=1)
    beyond = catalog.get_document("abc", page=3, page_size=1)

    assert first.total_pages == 2
    assert [chunk.chunk_id for chunk in first.chunks] == ["abc-p0001-c0000"]
    assert [chunk.chunk_id for chunk in second.chunks] == ["abc-p0001-c0001"]
    assert beyond.chunks == []


def test_catalog_reads_legacy_document_chunks(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    legacy = next(
        document
        for document in catalog.list_documents()
        if document.document_id.startswith("legacy-")
    )

    detail = catalog.get_document(legacy.document_id, page=1, page_size=20)

    assert detail.document.filename == "legacy.pdf"
    assert [chunk.page_number for chunk in detail.chunks] == [1, 2]


def test_catalog_rejects_unknown_document(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)

    with pytest.raises(DocumentNotFoundError):
        catalog.get_document("missing", page=1, page_size=20)
