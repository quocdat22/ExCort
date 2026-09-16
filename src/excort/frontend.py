"""Streamlit chat and document browser for the ExCort FastAPI backend."""

import math
from typing import TypedDict

import httpx
import streamlit as st

from excort.config import settings
from excort.schemas import (
    ChatResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    HealthResponse,
    SourceResponse,
)


class ChatMessage(TypedDict):
    """Serializable message shape stored in Streamlit session state."""

    role: str
    content: str
    sources: list[dict]


def get_backend_health() -> HealthResponse:
    """Fetch backend readiness for the sidebar status indicator."""
    response = httpx.get(f"{settings.backend_url}/health", timeout=3.0)
    response.raise_for_status()
    return HealthResponse.model_validate(response.json())


def ask_backend(question: str) -> ChatResponse:
    """Send one question to FastAPI and validate its response contract."""
    response = httpx.post(
        f"{settings.backend_url}/chat",
        json={"question": question, "top_k": settings.top_k},
        timeout=180.0,
    )
    response.raise_for_status()
    return ChatResponse.model_validate(response.json())


def upload_document(filename: str, content: bytes) -> DocumentUploadResponse:
    """Upload one PDF and validate the completed indexing summary."""
    response = httpx.post(
        f"{settings.backend_url}/documents",
        files={"file": (filename, content, "application/pdf")},
        timeout=300.0,
    )
    response.raise_for_status()
    return DocumentUploadResponse.model_validate(response.json())


def get_documents() -> DocumentListResponse:
    """Fetch and validate the indexed document list."""
    response = httpx.get(f"{settings.backend_url}/documents", timeout=10.0)
    response.raise_for_status()
    return DocumentListResponse.model_validate(response.json())


def get_document_detail(
    document_id: str,
    *,
    page: int,
    page_size: int,
) -> DocumentDetailResponse:
    """Fetch one page of chunks for an indexed document."""
    response = httpx.get(
        f"{settings.backend_url}/documents/{document_id}",
        params={"page": page, "page_size": page_size},
        timeout=10.0,
    )
    response.raise_for_status()
    return DocumentDetailResponse.model_validate(response.json())


def backend_error_message(error: httpx.HTTPError | ValueError) -> str:
    """Prefer the API's safe detail message over a verbose HTTP exception."""
    if isinstance(error, httpx.HTTPStatusError):
        try:
            detail = error.response.json().get("detail")
        except ValueError:
            detail = None
        if isinstance(detail, str) and detail:
            return detail
    return str(error)


def render_sources(sources: list[SourceResponse]) -> None:
    """Keep evidence available without overwhelming the main conversation."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(
                f"**Page {source.page_number}** · `{source.chunk_id}` · "
                f"similarity `{source.similarity:.3f}`"
            )
            st.caption(source.text)


def render_chat() -> None:
    """Render the stateful chat conversation and question input."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(
                [SourceResponse.model_validate(item) for item in message["sources"]]
            )

    question = st.chat_input("Ask about the document")
    if not question:
        return

    user_message: ChatMessage = {
        "role": "user",
        "content": question,
        "sources": [],
    }
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document and generating an answer..."):
                response = ask_backend(question)
            st.markdown(response.answer)
            render_sources(response.sources)
            assistant_message: ChatMessage = {
                "role": "assistant",
                "content": response.answer,
                "sources": [source.model_dump() for source in response.sources],
            }
        except (httpx.HTTPError, ValueError) as error:
            error_message = f"Unable to reach the chatbot backend: {error}"
            st.error(error_message)
            assistant_message = {
                "role": "assistant",
                "content": error_message,
                "sources": [],
            }
        st.session_state.messages.append(assistant_message)


def render_document_browser() -> None:
    """Render document selection and full chunk metadata with pagination."""
    try:
        result = get_documents()
    except (httpx.HTTPError, ValueError) as error:
        st.error(f"Unable to load documents: {backend_error_message(error)}")
        return
    if not result.documents:
        st.info("No indexed documents are available.")
        return

    by_id = {document.document_id: document for document in result.documents}
    document_id = st.selectbox(
        "Document",
        options=list(by_id),
        format_func=lambda item: (
            f"{by_id[item].filename} ({by_id[item].chunk_count} chunks)"
        ),
    )
    document = by_id[document_id]
    st.caption(
        f"{document.indexed_page_count} indexed pages · {document.chunk_count} chunks"
    )

    page_size = 20
    total_pages = max(1, math.ceil(document.chunk_count / page_size))
    page = int(
        st.number_input(
            "Chunk page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=f"document_page_{document_id}",
        )
    )
    try:
        detail = get_document_detail(
            document_id,
            page=page,
            page_size=page_size,
        )
    except (httpx.HTTPError, ValueError) as error:
        st.error(f"Unable to load document chunks: {backend_error_message(error)}")
        return

    st.caption(f"Chunk page {detail.page} of {detail.total_pages}")
    for chunk in detail.chunks:
        with st.expander(
            f"Page {chunk.page_number} · chunk {chunk.chunk_index} · "
            f"{chunk.token_count} tokens"
        ):
            st.caption(
                f"ID: {chunk.chunk_id} · tokens {chunk.token_start}–{chunk.token_end}"
            )
            st.text(chunk.text)


def render_app() -> None:
    """Render chat, upload controls, and the indexed document browser."""
    st.set_page_config(page_title="ExCort", page_icon="📚")
    st.title("ExCort")
    st.caption("Ask questions grounded in the indexed PDF document.")

    with st.sidebar:
        st.header("System status")
        uploaded_file = st.file_uploader(
            "Add a PDF document",
            type=["pdf"],
            help=f"Text-based PDF, up to {settings.max_upload_size_mb} MB.",
        )
        if st.button(
            "Upload & index",
            disabled=uploaded_file is None,
            use_container_width=True,
        ):
            try:
                with st.spinner("Parsing, embedding, and indexing the PDF..."):
                    uploaded = upload_document(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                    )
                st.success(
                    f"Indexed {uploaded.filename}: "
                    f"{uploaded.indexed_page_count} pages, "
                    f"{uploaded.chunk_count} chunks"
                )
            except (httpx.HTTPError, ValueError) as error:
                st.error(f"Upload failed: {backend_error_message(error)}")

        try:
            health = get_backend_health()
            st.success(f"Backend ready · {health.record_count} chunks")
        except (httpx.HTTPError, ValueError):
            st.warning("Backend unavailable. Start FastAPI on the configured URL.")

    chat_tab, documents_tab = st.tabs(["Chat", "Documents"])
    with chat_tab:
        render_chat()
    with documents_tab:
        render_document_browser()


if __name__ == "__main__":
    render_app()
