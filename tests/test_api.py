"""FastAPI endpoint tests with the external RAG calls mocked."""

from fastapi.testclient import TestClient

from excort.api import app, get_document_indexer, get_rag_chain
from excort.generation import FinalPrompt
from excort.indexing import IndexedDocument, UploadTooLargeError
from excort.rag import RagResponse
from excort.retrieval import RetrievedChunk


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
