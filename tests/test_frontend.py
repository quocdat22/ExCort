"""Tests for frontend communication helpers used by the upload UI."""

import httpx
import pytest

from excort.frontend import backend_error_message, upload_document


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
