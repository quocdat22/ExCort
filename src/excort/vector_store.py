"""Persistent Chroma storage for dense vectors and chunk metadata."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from excort.ingestion import Chunk


class DocumentNotFoundError(LookupError):
    """The requested document has no chunks in the collection."""


@dataclass(frozen=True)
class StoredDocument:
    """A document summary derived from its indexed chunk metadata."""

    document_id: str
    filename: str
    indexed_page_count: int
    chunk_count: int


@dataclass(frozen=True)
class StoredChunk:
    """One stored chunk and the metadata needed to inspect it."""

    chunk_id: str
    page_number: int
    chunk_index: int
    token_start: int
    token_end: int
    token_count: int
    text: str


@dataclass(frozen=True)
class StoredDocumentPage:
    """A deterministic page of chunks belonging to one document."""

    document: StoredDocument
    page: int
    page_size: int
    total_pages: int
    chunks: list[StoredChunk]


@dataclass
class _DocumentGroup:
    """Internal aggregation state for records from the same document."""

    document_id: str
    source: str
    page_numbers: set[int]
    chunk_count: int = 0


class DocumentCatalog:
    """Read document summaries and inspect chunks without loading embeddings."""

    def __init__(self, persist_path: Path, collection_name: str) -> None:
        self.persist_path = persist_path
        self.collection_name = collection_name

    def list_documents(self) -> list[StoredDocument]:
        """Group collection records into stable, human-readable summaries."""
        collection = get_collection(self.persist_path, self.collection_name)
        result = collection.get(include=["metadatas"])
        ids = result["ids"]
        metadatas = result["metadatas"] or []
        if len(ids) != len(metadatas):
            raise RuntimeError("Vector collection returned inconsistent records")

        groups: dict[str, _DocumentGroup] = {}
        for metadata in metadatas:
            if metadata is None:
                raise RuntimeError("Vector collection contains missing metadata")
            source = str(metadata["source"])
            document_id = self._document_id(metadata, source)
            group = groups.setdefault(
                document_id,
                _DocumentGroup(
                    document_id=document_id,
                    source=source,
                    page_numbers=set(),
                ),
            )
            group.page_numbers.add(int(metadata["page_number"]))
            group.chunk_count += 1

        documents = [
            StoredDocument(
                document_id=group.document_id,
                filename=Path(group.source).name,
                indexed_page_count=len(group.page_numbers),
                chunk_count=group.chunk_count,
            )
            for group in groups.values()
        ]
        return sorted(
            documents,
            key=lambda document: (document.filename.casefold(), document.document_id),
        )

    def get_document(
        self,
        document_id: str,
        *,
        page: int,
        page_size: int,
    ) -> StoredDocumentPage:
        """Return one page of chunks ordered by page and chunk position."""
        documents = {item.document_id: item for item in self.list_documents()}
        document = documents.get(document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)

        collection = get_collection(self.persist_path, self.collection_name)
        if document_id.startswith("legacy-"):
            source = self._legacy_source(collection, document_id)
            where = {"source": source}
        else:
            where = {"document_id": document_id}
        result = collection.get(
            where=where,
            include=["documents", "metadatas"],
        )
        ids = result["ids"]
        texts = result["documents"] or []
        metadatas = result["metadatas"] or []
        if not (len(ids) == len(texts) == len(metadatas)):
            raise RuntimeError("Vector collection returned inconsistent records")

        chunks = [
            StoredChunk(
                chunk_id=chunk_id,
                page_number=int(metadata["page_number"]),
                chunk_index=int(metadata["chunk_index"]),
                token_start=int(metadata["token_start"]),
                token_end=int(metadata["token_end"]),
                token_count=int(metadata["token_count"]),
                text=text,
            )
            for chunk_id, text, metadata in zip(ids, texts, metadatas, strict=True)
            if metadata is not None
        ]
        if len(chunks) != len(ids):
            raise RuntimeError("Vector collection contains missing metadata")
        chunks.sort(
            key=lambda chunk: (
                chunk.page_number,
                chunk.chunk_index,
                chunk.chunk_id,
            )
        )
        start = (page - 1) * page_size
        total_pages = math.ceil(len(chunks) / page_size)
        return StoredDocumentPage(
            document=document,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            chunks=chunks[start : start + page_size],
        )

    @staticmethod
    def _document_id(metadata: dict, source: str) -> str:
        stored_id = metadata.get("document_id")
        if isinstance(stored_id, str) and stored_id:
            return stored_id
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        return f"legacy-{digest}"

    def _legacy_source(self, collection: Collection, document_id: str) -> str:
        result = collection.get(include=["metadatas"])
        for metadata in result["metadatas"] or []:
            if metadata is None:
                continue
            source = str(metadata["source"])
            if self._document_id(metadata, source) == document_id:
                return source
        raise DocumentNotFoundError(document_id)


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
