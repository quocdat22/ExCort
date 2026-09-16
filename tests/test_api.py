"""FastAPI endpoint tests with the external RAG calls mocked."""

from fastapi.testclient import TestClient

from excort.api import (
    app,
    get_document_catalog,
    get_document_indexer,
    get_rag_chain,
)
from excort.generation import FinalPrompt
from excort.indexing import IndexedDocument, UploadTooLargeError
from excort.rag import RagResponse
from excort.retrieval import RetrievedChunk
from excort.vector_store import (
    DocumentNotFoundError,
    StoredChunk,
    StoredDocument,
    StoredDocumentPage,
)


class FakeRagChain:
    def answer(self, question: str, top_k: int) -> RagResponse:
        assert question == "What is MLOps?"
        assert top_k == 2
        return RagResponse(
            answer="MLOps brings ML into production [page 2].",
            sources=[
                RetrievedChunk(
                    id="demo-p0002-c0000",
                    text="MLOps context",
                    metadata={"source": "demo.pdf", "page_number": 2},
                    similarity=0.9,
                )
            ],
            prompt=FinalPrompt(system="system", user="user"),
        )


def test_chat_endpoint_returns_validated_grounded_response() -> None:
    app.dependency_overrides[get_rag_chain] = lambda: FakeRagChain()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/chat", json={"question": "What is MLOps?", "top_k": 2}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].endswith("[page 2].")
    assert payload["sources"][0]["page_number"] == 2


def test_chat_endpoint_rejects_whitespace_question() -> None:
    app.dependency_overrides[get_rag_chain] = lambda: FakeRagChain()
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"question": "   ", "top_k": 2})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 422


class FakeDocumentIndexer:
    def index(self, stream, filename: str | None) -> IndexedDocument:
        assert filename == "guide.pdf"
        assert stream.read() == b"%PDF-test"
        return IndexedDocument(
            document_id="a" * 64,
            filename="guide.pdf",
            size_bytes=9,
            indexed_page_count=2,
            chunk_count=3,
        )


def test_upload_endpoint_returns_indexing_summary() -> None:
    app.dependency_overrides[get_document_indexer] = lambda: FakeDocumentIndexer()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/documents",
                files={"file": ("guide.pdf", b"%PDF-test", "application/pdf")},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["filename"] == "guide.pdf"
    assert response.json()["chunk_count"] == 3


class OversizeDocumentIndexer:
    def index(self, stream, filename: str | None) -> IndexedDocument:
        raise UploadTooLargeError("PDF exceeds the configured limit")


def test_upload_endpoint_maps_oversize_error_to_413() -> None:
    app.dependency_overrides[get_document_indexer] = lambda: OversizeDocumentIndexer()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/documents",
                files={"file": ("large.pdf", b"content", "application/pdf")},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 413
    assert response.json()["detail"] == "PDF exceeds the configured limit"


class FakeDocumentCatalog:
    document = StoredDocument(
        document_id="doc-1",
        filename="guide.pdf",
        indexed_page_count=2,
        chunk_count=3,
    )

    def list_documents(self) -> list[StoredDocument]:
        return [self.document]

    def get_document(
        self,
        document_id: str,
        *,
        page: int,
        page_size: int,
    ) -> StoredDocumentPage:
        if document_id != self.document.document_id:
            raise DocumentNotFoundError(document_id)
        assert page_size == 1
        return StoredDocumentPage(
            document=self.document,
            page=page,
            page_size=page_size,
            total_pages=3,
            chunks=[
                StoredChunk(
                    chunk_id="doc-1-p0001-c0000",
                    page_number=1,
                    chunk_index=0,
                    token_start=0,
                    token_end=10,
                    token_count=10,
                    text="Inspectable chunk",
                )
            ],
        )


def test_list_documents_endpoint_returns_summaries() -> None:
    app.dependency_overrides[get_document_catalog] = lambda: FakeDocumentCatalog()
    try:
        with TestClient(app) as client:
            response = client.get("/documents")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "total": 1,
        "documents": [
            {
                "document_id": "doc-1",
                "filename": "guide.pdf",
                "indexed_page_count": 2,
                "chunk_count": 3,
            }
        ],
    }


def test_document_detail_endpoint_returns_paginated_chunks() -> None:
    app.dependency_overrides[get_document_catalog] = lambda: FakeDocumentCatalog()
    try:
        with TestClient(app) as client:
            response = client.get("/documents/doc-1?page=2&page_size=1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["page"] == 2
    assert payload["total_pages"] == 3
    assert payload["chunks"][0]["text"] == "Inspectable chunk"


def test_document_detail_endpoint_returns_404_for_unknown_id() -> None:
    app.dependency_overrides[get_document_catalog] = lambda: FakeDocumentCatalog()
    try:
        with TestClient(app) as client:
            response = client.get("/documents/missing?page_size=1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_document_detail_endpoint_validates_pagination() -> None:
    app.dependency_overrides[get_document_catalog] = lambda: FakeDocumentCatalog()
    try:
        with TestClient(app) as client:
            response = client.get("/documents/doc-1?page=0&page_size=101")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


class UnavailableDocumentCatalog:
    def list_documents(self) -> list[StoredDocument]:
        raise RuntimeError("collection unavailable")


def test_list_documents_endpoint_maps_collection_error_to_503() -> None:
    app.dependency_overrides[get_document_catalog] = lambda: (
        UnavailableDocumentCatalog()
    )
    try:
        with TestClient(app) as client:
            response = client.get("/documents")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["detail"].startswith("Vector collection is unavailable")
