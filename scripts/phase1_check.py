"""Run ingestion and print observable chunking results."""

from pathlib import Path

import tiktoken

from excort.config import settings
from excort.ingestion import ENCODING_NAME, ingest_pdf, parse_pdf

OUTPUT_PATH = Path("data/processed/chunks.json")


def main() -> None:
    pages = parse_pdf(settings.pdf_path)
    chunks = ingest_pdf(
        pdf_path=settings.pdf_path,
        output_path=OUTPUT_PATH,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    print(f"PDF: {settings.pdf_path}")
    print(f"Pages with extractable text: {len(pages)}")
    print(f"Encoding: {ENCODING_NAME}")
    print(f"Chunk size / overlap: {settings.chunk_size} / {settings.chunk_overlap}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("Token count for every chunk:")
    for chunk in chunks:
        print(f"  {chunk.id}: {chunk.token_count}")

    print("\nSample chunks:")
    for chunk in chunks[:2]:
        sample = chunk.text[:500].replace("\n", " ")
        print(
            f"\n[{chunk.id}] page={chunk.page_number}, "
            f"tokens={chunk.token_count}\n{sample}"
        )

    encoding = tiktoken.get_encoding(ENCODING_NAME)
    invalid_chunks = [
        chunk
        for chunk in chunks
        if chunk.token_count > settings.chunk_size
        or len(encoding.encode(chunk.text)) != chunk.token_count
    ]
    if not chunks or invalid_chunks:
        raise SystemExit("Phase 1 check failed: invalid chunk output")
    print("\nPhase 1 check: PASS")


if __name__ == "__main__":
    main()
