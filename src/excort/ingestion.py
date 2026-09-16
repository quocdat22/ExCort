"""PDF parsing and token-based fixed-size chunking for ExCort."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
import tiktoken

ENCODING_NAME = "cl100k_base"


@dataclass(frozen=True)
class PageText:
    """Text extracted from one PDF page, using a human-readable page number."""

    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable text unit and the metadata needed to trace its origin."""

    id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    token_start: int
    token_end: int
    token_count: int
    document_id: str | None = None


def parse_pdf(pdf_path: Path) -> list[PageText]:
    """Extract plain text page-by-page from a PDF with PyMuPDF.

    Page boundaries are retained so a retrieved answer can later cite its source.
    OCR is intentionally out of scope for this PDF-only MVP.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[PageText] = []
    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError(f"Password-protected PDFs are not supported: {pdf_path}")

        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()
            if text:
                pages.append(PageText(page_number=page_index + 1, text=text))

    if not pages:
        raise ValueError(f"No extractable text found in PDF: {pdf_path}")
    return pages


def chunk_pages(
    pages: list[PageText],
    source: Path,
    chunk_size: int,
    chunk_overlap: int,
    document_id: str | None = None,
) -> list[Chunk]:
    """Split each page into fixed token windows with overlap.

    Tokens, rather than characters, make chunk size predictable for embedding and
    LLM APIs. Chunks do not cross pages so their page metadata remains exact.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size")

    encoding = tiktoken.get_encoding(ENCODING_NAME)
    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []

    for page in pages:
        token_ids = encoding.encode(page.text)
        page_chunk_index = 0
        for token_start in range(0, len(token_ids), step):
            token_end = min(token_start + chunk_size, len(token_ids))
            chunk_token_ids = token_ids[token_start:token_end]
            id_prefix = document_id or source.stem
            chunk_id = f"{id_prefix}-p{page.page_number:04d}-c{page_chunk_index:04d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    # Do not strip here: whitespace is tokenized too, so retaining it
                    # keeps token_count exactly reproducible from the stored text.
                    text=encoding.decode(chunk_token_ids),
                    source=str(source),
                    page_number=page.page_number,
                    chunk_index=page_chunk_index,
                    token_start=token_start,
                    token_end=token_end,
                    token_count=len(chunk_token_ids),
                    document_id=document_id,
                )
            )
            page_chunk_index += 1
            if token_end == len(token_ids):
                break

    return chunks


def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
    """Persist chunks as readable JSON for inspection and later embedding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ingest_pdf(
    pdf_path: Path,
    output_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Run the complete local ingestion phase and return its chunks."""
    pages = parse_pdf(pdf_path)
    chunks = chunk_pages(pages, pdf_path, chunk_size, chunk_overlap)
    save_chunks(chunks, output_path)
    return chunks
