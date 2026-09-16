"""Persistent Chroma storage for dense vectors and chunk metadata."""

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from excort.ingestion import Chunk


def rebuild_collection(
    persist_path: Path,
    collection_name: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> Collection:
    """Replace the generated collection so re-indexing cannot leave stale chunks."""
    if not chunks:
        raise ValueError("cannot index an empty chunk list")
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    try:
        client.delete_collection(collection_name)
    except (NotFoundError, ValueError):
        pass

    # Cosine distance matches the normalized retrieval vectors requested from Jina.
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "token_start": chunk.token_start,
                "token_end": chunk.token_end,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ],
    )
    return collection


def get_or_create_collection(persist_path: Path, collection_name: str) -> Collection:
    """Open the shared collection, creating it with cosine distance if needed."""
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def document_exists(persist_path: Path, collection_name: str, document_id: str) -> bool:
    """Check whether any indexed chunk belongs to a content-addressed document."""
    try:
        collection = get_collection(persist_path, collection_name)
    except (NotFoundError, ValueError):
        return False
    result = collection.get(
        where={"document_id": document_id},
        limit=1,
        include=[],
    )
    return bool(result["ids"])


def add_document(
    persist_path: Path,
    collection_name: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> Collection:
    """Add one document without replacing records belonging to earlier uploads."""
    if not chunks:
        raise ValueError("cannot index an empty chunk list")
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    if any(chunk.document_id is None for chunk in chunks):
        raise ValueError("uploaded chunks must include a document_id")

    collection = get_or_create_collection(persist_path, collection_name)
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "token_start": chunk.token_start,
                "token_end": chunk.token_end,
                "token_count": chunk.token_count,
                "document_id": chunk.document_id or "",
            }
            for chunk in chunks
        ],
    )
    return collection


def get_collection(persist_path: Path, collection_name: str) -> Collection:
    """Open an existing persistent Chroma collection."""
    client = chromadb.PersistentClient(path=str(persist_path))
    return client.get_collection(collection_name)
