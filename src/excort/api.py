"""FastAPI backend exposing health and RAG chat endpoints."""

from functools import lru_cache
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, status

from excort.config import settings
from excort.embeddings import JinaEmbeddingClient
from excort.generation import OpenRouterClient
from excort.rag import RagChain
from excort.retrieval import DenseRetriever
from excort.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SourceResponse,
)
from excort.vector_store import get_collection

app = FastAPI(
    title="ExCort API",
    description="PDF-grounded dense-retrieval chatbot API",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_rag_chain() -> RagChain:
    """Build shared stateless clients once instead of once per request."""
    if not settings.jina_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JINA_API_KEY is missing from .env",
        )
    if not settings.openrouter_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENROUTER_API_KEY is missing from .env",
        )

    embedder = JinaEmbeddingClient(
        api_key=settings.jina_api_key,
        model=settings.jina_embedding_model,
    )
    try:
        retriever = DenseRetriever(
            embedder=embedder,
            persist_path=settings.chroma_path,
            collection_name=settings.chroma_collection,
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector collection is unavailable; run Phase 2 first",
        ) from error
    llm = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
    )
    return RagChain(retriever=retriever, llm=llm)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Confirm that the API can open the generated Chroma collection."""
    try:
        collection = get_collection(settings.chroma_path, settings.chroma_collection)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector collection is unavailable; run Phase 2 first",
        ) from error
    return HealthResponse(
        status="ok",
        collection=settings.chroma_collection,
        record_count=collection.count(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag_chain: Annotated[RagChain, Depends(get_rag_chain)],
) -> ChatResponse:
    """Retrieve PDF evidence and generate one grounded response.

    This is deliberately synchronous: FastAPI runs normal ``def`` endpoints in a
    thread pool, keeping blocking Jina/OpenRouter HTTP calls off the event loop.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="question must contain non-whitespace characters",
        )

    try:
        result = rag_chain.answer(question, top_k=request.top_k or settings.top_k)
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream model service returned an error",
        ) from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    return ChatResponse(
        question=question,
        answer=result.answer,
        sources=[
            SourceResponse(
                chunk_id=source.id,
                source=str(source.metadata["source"]),
                page_number=int(source.metadata["page_number"]),
                similarity=source.similarity,
                text=source.text,
            )
            for source in result.sources
        ],
    )
