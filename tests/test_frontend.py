"""Tests for frontend communication helpers used by the upload UI."""

import httpx
import pytest

from excort.frontend import (
    backend_error_message,
    get_document_detail,
    get_documents,
    upload_document,
)


def test_upload_document_posts_pdf_and_validates_response(monkeypatch) -> None:
    def fake_post(url: str, **kwargs) -> httpx.Response:
        filename, content, content_type = kwargs["files"]["file"]
        assert url.endswith("/documents")
        assert (filename, content, content_type) == (
            "guide.pdf",
            b"pdf-content",
            "application/pdf",
        )
        return httpx.Response(
            201,
            request=httpx.Request("POST", url),
            json={
                "document_id": "a" * 64,
                "filename": "guide.pdf",
                "size_bytes": 11,
                "indexed_page_count": 2,
                "chunk_count": 3,
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    result = upload_document("guide.pdf", b"pdf-content")

    assert result.filename == "guide.pdf"
    assert result.chunk_count == 3


def test_backend_error_message_uses_api_detail() -> None:
    response = httpx.Response(
        409,
        request=httpx.Request("POST", "http://backend/documents"),
        json={"detail": "This PDF is already indexed"},
    )
    with pytest.raises(httpx.HTTPStatusError) as captured:
        response.raise_for_status()

    assert backend_error_message(captured.value) == "This PDF is already indexed"


def test_get_documents_validates_document_summaries(monkeypatch) -> None:
    def fake_get(url: str, **kwargs) -> httpx.Response:
        assert url.endswith("/documents")
        assert kwargs["timeout"] == 10.0
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "total": 1,
                "documents": [
                    {
                        "document_id": "doc-1",
                        "filename": "guide.pdf",
                        "indexed_page_count": 2,
                        "chunk_count": 3,
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    result = get_documents()

    assert result.total == 1
    assert result.documents[0].filename == "guide.pdf"


def test_get_document_detail_sends_pagination_and_validates_chunks(
    monkeypatch,
) -> None:
    def fake_get(url: str, **kwargs) -> httpx.Response:
        assert url.endswith("/documents/doc-1")
        assert kwargs["params"] == {"page": 2, "page_size": 20}
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "document_id": "doc-1",
                "filename": "guide.pdf",
                "indexed_page_count": 2,
                "chunk_count": 21,
                "page": 2,
                "page_size": 20,
                "total_pages": 2,
                "chunks": [
                    {
                        "chunk_id": "doc-1-p0002-c0000",
                        "page_number": 2,
                        "chunk_index": 0,
                        "token_start": 0,
                        "token_end": 10,
                        "token_count": 10,
                        "text": "Chunk content",
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    result = get_document_detail("doc-1", page=2, page_size=20)

    assert result.page == 2
    assert result.chunks[0].text == "Chunk content"
