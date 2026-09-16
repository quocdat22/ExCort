"""Tests for validated, additive PDF upload indexing."""

from io import BytesIO
from pathlib import Path

import pymupdf
import pytest

import excort.indexing
from excort.indexing import (
    DocumentIndexer,
    DuplicateDocumentError,
    InvalidPdfError,
    TextlessPdfError,
    UnsupportedPdfError,
    UploadTooLargeError,
)
from excort.vector_store import get_collection


class FakeEmbedder:
    """Return deterministic vectors without making external requests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 1.0] for index, _text in enumerate(texts)]


def make_pdf(text: str | None) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    if text is not None:
        page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def make_mixed_pdf() -> bytes:
    document = pymupdf.open()
    document.new_page()
    text_page = document.new_page()
    text_page.insert_text((72, 72), "Extractable page")
    content = document.tobytes()
    document.close()
    return content


def make_indexer(tmp_path: Path, max_size_bytes: int = 1024 * 1024) -> DocumentIndexer:
    return DocumentIndexer(
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        persist_path=tmp_path / "chroma",
        collection_name="test_documents",
        chunk_size=50,
        chunk_overlap=5,
        max_size_bytes=max_size_bytes,
        upload_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )


def test_indexer_appends_documents_and_preserves_source_name(tmp_path: Path) -> None:
    indexer = make_indexer(tmp_path)

    first = indexer.index(BytesIO(make_pdf("First document")), "first.pdf")
    second = indexer.index(BytesIO(make_pdf("Second document")), "second.pdf")

    collection = get_collection(tmp_path / "chroma", "test_documents")
    result = collection.get(include=["metadatas"])
    assert collection.count() == first.chunk_count + second.chunk_count
    assert {metadata["source"] for metadata in result["metadatas"]} == {
        "first.pdf",
        "second.pdf",
    }
    assert (tmp_path / "raw" / f"{first.document_id}.pdf").is_file()
    assert (tmp_path / "processed" / f"{second.document_id}.json").is_file()


def test_indexer_rejects_duplicate_content(tmp_path: Path) -> None:
    indexer = make_indexer(tmp_path)
    pdf = make_pdf("Same content")
    indexer.index(BytesIO(pdf), "original.pdf")

    with pytest.raises(DuplicateDocumentError):
        indexer.index(BytesIO(pdf), "renamed.pdf")


def test_indexer_rejects_file_over_size_limit(tmp_path: Path) -> None:
    indexer = make_indexer(tmp_path, max_size_bytes=4)

    with pytest.raises(UploadTooLargeError):
        indexer.index(BytesIO(b"%PDF-1.7"), "large.pdf")


def test_indexer_accepts_file_exactly_at_size_limit(tmp_path: Path) -> None:
    pdf = make_pdf("Boundary content")
    result = make_indexer(tmp_path, max_size_bytes=len(pdf)).index(
        BytesIO(pdf), "boundary.pdf"
    )

    assert result.size_bytes == len(pdf)


def test_indexer_rejects_non_pdf_extension(tmp_path: Path) -> None:
    with pytest.raises(InvalidPdfError):
        make_indexer(tmp_path).index(BytesIO(make_pdf("text")), "document.txt")


def test_indexer_rejects_pdf_without_text_layer(tmp_path: Path) -> None:
    with pytest.raises(TextlessPdfError):
        make_indexer(tmp_path).index(BytesIO(make_pdf(None)), "scan.pdf")


def test_indexer_accepts_mixed_pdf_and_skips_blank_pages(tmp_path: Path) -> None:
    result = make_indexer(tmp_path).index(BytesIO(make_mixed_pdf()), "mixed.pdf")

    assert result.indexed_page_count == 1


def test_indexer_rejects_password_protected_pdf(tmp_path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Protected text")
    content = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="secret",
    )
    document.close()

    with pytest.raises(UnsupportedPdfError):
        make_indexer(tmp_path).index(BytesIO(content), "protected.pdf")


def test_indexer_sanitizes_source_filename(tmp_path: Path) -> None:
    indexer = make_indexer(tmp_path)
    result = indexer.index(BytesIO(make_pdf("Safe source")), "../../unsafe.pdf")

    collection = get_collection(tmp_path / "chroma", "test_documents")
    metadata = collection.get(include=["metadatas"])["metadatas"][0]
    assert result.filename == "unsafe.pdf"
    assert metadata["source"] == "unsafe.pdf"


def test_indexer_removes_artifacts_when_vector_write_fails(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_add_document(*args, **kwargs) -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(excort.indexing, "add_document", fail_add_document)

    with pytest.raises(RuntimeError, match="database unavailable"):
        make_indexer(tmp_path).index(BytesIO(make_pdf("Rollback")), "rollback.pdf")

    assert list((tmp_path / "raw").iterdir()) == []
    assert list((tmp_path / "processed").iterdir()) == []
