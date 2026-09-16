"""Streamlit chat interface for the ExCort FastAPI backend."""

from typing import TypedDict

import httpx
import streamlit as st

from excort.config import settings
from excort.schemas import (
    ChatResponse,
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


def render_app() -> None:
    """Render the complete chat UI and preserve messages across reruns."""
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


if __name__ == "__main__":
    render_app()
