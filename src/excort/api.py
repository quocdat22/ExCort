"""FastAPI backend exposing health and RAG chat endpoints."""

from functools import lru_cache
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status

from excort.config import settings
from excort.embeddings import JinaEmbeddingClient
from excort.generation import OpenRouterClient
from excort.indexing import (
    DocumentIndexer,
    DuplicateDocumentError,
    EmbeddingServiceError,
    InvalidPdfError,
    TextlessPdfError,
    UnsupportedPdfError,
    UploadTooLargeError,
)
from excort.rag import RagChain
from excort.retrieval import DenseRetriever
from excort.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentChunkResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummaryResponse,
    DocumentUploadResponse,
    HealthResponse,
    SourceResponse,
)
from excort.vector_store import (
    DocumentCatalog,
    DocumentNotFoundError,
    get_collection,
)

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


@lru_cache(maxsize=1)
def get_document_indexer() -> DocumentIndexer:
    """Build the shared upload pipeline and serialize its index mutations."""
    if not settings.jina_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JINA_API_KEY is missing from .env",
        )
    return DocumentIndexer(
        embedder=JinaEmbeddingClient(
            api_key=settings.jina_api_key,
            model=settings.jina_embedding_model,
        ),
        persist_path=settings.chroma_path,
        collection_name=settings.chroma_collection,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_size_bytes=settings.max_upload_size_mb * 1024 * 1024,
    )


def get_document_catalog() -> DocumentCatalog:
    """Build the read-only view over documents stored in Chroma."""
    return DocumentCatalog(settings.chroma_path, settings.chroma_collection)


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


@app.get("/documents", response_model=DocumentListResponse)
def list_documents(
    catalog: Annotated[DocumentCatalog, Depends(get_document_catalog)],
) -> DocumentListResponse:
    """List documents represented by chunks in the vector collection."""
    try:
        documents = catalog.list_documents()
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector collection is unavailable; run Phase 2 first",
        ) from error
    summaries = [
        DocumentSummaryResponse(
            document_id=document.document_id,
            filename=document.filename,
            indexed_page_count=document.indexed_page_count,
            chunk_count=document.chunk_count,
        )
        for document in documents
    ]
    return DocumentListResponse(total=len(summaries), documents=summaries)


@app.get("/documents/{document_id}", response_model=DocumentDetailResponse)
def get_document(
    document_id: str,
    catalog: Annotated[DocumentCatalog, Depends(get_document_catalog)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> DocumentDetailResponse:
    """Return an ordered, paginated view of one document's chunks."""
    try:
        result = catalog.get_document(
            document_id,
            page=page,
            page_size=page_size,
        )
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector collection is unavailable; run Phase 2 first",
        ) from error
    return DocumentDetailResponse(
        document_id=result.document.document_id,
        filename=result.document.filename,
        indexed_page_count=result.document.indexed_page_count,
        chunk_count=result.document.chunk_count,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
        chunks=[
            DocumentChunkResponse(
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                token_start=chunk.token_start,
                token_end=chunk.token_end,
                token_count=chunk.token_count,
                text=chunk.text,
            )
            for chunk in result.chunks
        ],
    )


@app.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: Annotated[UploadFile, File(description="Text-based PDF to index")],
    indexer: Annotated[DocumentIndexer, Depends(get_document_indexer)],
) -> DocumentUploadResponse:
    """Validate and append one uploaded PDF to the shared vector collection."""
    try:
        document = indexer.index(file.file, file.filename)
    except UploadTooLargeError as error:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(error),
        ) from error
    except InvalidPdfError as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(error),
        ) from error
    except (TextlessPdfError, UnsupportedPdfError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except DuplicateDocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except EmbeddingServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to persist the uploaded document",
        ) from error

    return DocumentUploadResponse(
        document_id=document.document_id,
        filename=document.filename,
        size_bytes=document.size_bytes,
        indexed_page_count=document.indexed_page_count,
        chunk_count=document.chunk_count,
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
