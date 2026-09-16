"""Validated, content-addressed PDF upload and indexing orchestration."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import httpx
import pymupdf

from excort.embeddings import JinaEmbeddingClient
from excort.ingestion import chunk_pages, parse_pdf, save_chunks
from excort.vector_store import add_document, document_exists

READ_SIZE = 64 * 1024


class InvalidPdfError(ValueError):
    """The upload is not a supported, parseable PDF."""


class TextlessPdfError(ValueError):
    """The PDF has no extractable text layer."""


class UnsupportedPdfError(ValueError):
    """The PDF is valid but requires unsupported processing."""


class UploadTooLargeError(ValueError):
    """The upload exceeds the configured byte limit."""


class DuplicateDocumentError(ValueError):
    """The same PDF content is already indexed."""


class EmbeddingServiceError(RuntimeError):
    """The embedding provider failed while indexing the document."""


@dataclass(frozen=True)
class IndexedDocument:
    """Observable result of one completed indexing operation."""

    document_id: str
    filename: str
    size_bytes: int
    indexed_page_count: int
    chunk_count: int


class DocumentIndexer:
    """Validate, persist, embed, and append one uploaded PDF at a time."""

    def __init__(
        self,
        *,
        embedder: JinaEmbeddingClient,
        persist_path: Path,
        collection_name: str,
        chunk_size: int,
        chunk_overlap: int,
        max_size_bytes: int,
        upload_dir: Path = Path("data/raw/uploads"),
        processed_dir: Path = Path("data/processed/uploads"),
    ) -> None:
        self.embedder = embedder
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_size_bytes = max_size_bytes
        self.upload_dir = upload_dir
        self.processed_dir = processed_dir
        self._lock = threading.Lock()

    def index(self, stream: BinaryIO, filename: str | None) -> IndexedDocument:
        """Index a PDF and leave no new artifacts when an operation fails."""
        safe_name = Path(filename or "").name
        if not safe_name or Path(safe_name).suffix.lower() != ".pdf":
            raise InvalidPdfError("Only files with a .pdf extension are supported")

        temporary_pdf = self._copy_to_temporary_file(stream)
        temporary_chunks: Path | None = None
        stored_pdf: Path | None = None
        stored_chunks: Path | None = None
        try:
            document_id = self._sha256(temporary_pdf)
            self._validate_pdf_signature(temporary_pdf)
            try:
                pages = parse_pdf(temporary_pdf)
            except ValueError as error:
                message = str(error)
                if "No extractable text" in message:
                    raise TextlessPdfError(
                        "PDF has no extractable text; scanned PDFs require OCR"
                    ) from error
                if "Password-protected" in message:
                    raise UnsupportedPdfError(
                        "Password-protected PDFs are not supported"
                    ) from error
                raise InvalidPdfError(message) from error
            except (pymupdf.FileDataError, pymupdf.EmptyFileError) as error:
                raise InvalidPdfError("The uploaded file is not a valid PDF") from error

            chunks = chunk_pages(
                pages,
                Path(safe_name),
                self.chunk_size,
                self.chunk_overlap,
                document_id=document_id,
            )
            with self._lock:
                if document_exists(
                    self.persist_path, self.collection_name, document_id
                ):
                    raise DuplicateDocumentError("This PDF is already indexed")
                try:
                    embeddings = self.embedder.embed_documents(
                        [chunk.text for chunk in chunks]
                    )
                except (httpx.HTTPError, ValueError) as error:
                    raise EmbeddingServiceError(
                        "Embedding service returned an error"
                    ) from error

                self.upload_dir.mkdir(parents=True, exist_ok=True)
                self.processed_dir.mkdir(parents=True, exist_ok=True)
                stored_pdf = self.upload_dir / f"{document_id}.pdf"
                stored_chunks = self.processed_dir / f"{document_id}.json"
                temporary_chunks = self.processed_dir / f".{document_id}.json.tmp"
                try:
                    save_chunks(chunks, temporary_chunks)
                    # The system temp directory may be on another filesystem.
                    shutil.move(str(temporary_pdf), stored_pdf)
                    temporary_chunks.replace(stored_chunks)
                    add_document(
                        self.persist_path,
                        self.collection_name,
                        chunks,
                        embeddings,
                    )
                except Exception:
                    stored_pdf.unlink(missing_ok=True)
                    stored_chunks.unlink(missing_ok=True)
                    raise

            return IndexedDocument(
                document_id=document_id,
                filename=safe_name,
                size_bytes=stored_pdf.stat().st_size,
                indexed_page_count=len(pages),
                chunk_count=len(chunks),
            )
        finally:
            temporary_pdf.unlink(missing_ok=True)
            if temporary_chunks is not None:
                temporary_chunks.unlink(missing_ok=True)

    def _copy_to_temporary_file(self, stream: BinaryIO) -> Path:
        size = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary:
            temporary_path = Path(temporary.name)
            try:
                while data := stream.read(READ_SIZE):
                    size += len(data)
                    if size > self.max_size_bytes:
                        raise UploadTooLargeError(
                            f"PDF exceeds the {self.max_size_bytes} byte limit"
                        )
                    temporary.write(data)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        if size == 0:
            temporary_path.unlink(missing_ok=True)
            raise InvalidPdfError("Uploaded PDF is empty")
        return temporary_path

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while data := file.read(READ_SIZE):
                digest.update(data)
        return digest.hexdigest()

    @staticmethod
    def _validate_pdf_signature(path: Path) -> None:
        with path.open("rb") as file:
            header = file.read(1024)
        if b"%PDF-" not in header:
            raise InvalidPdfError("The uploaded file is not a valid PDF")
