"""Tests for deterministic token-based chunk construction."""

from pathlib import Path

import pytest
import tiktoken

from excort.ingestion import ENCODING_NAME, PageText, chunk_pages


def test_chunk_pages_uses_fixed_token_windows_with_overlap() -> None:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    text = "Machine learning systems need monitoring. " * 20
    token_ids = encoding.encode(text)

    chunks = chunk_pages(
        [PageText(page_number=3, text=text)],
        Path("data/raw/demo.pdf"),
        chunk_size=20,
        chunk_overlap=5,
    )

    assert chunks[0].id == "demo-p0003-c0000"
    assert chunks[0].token_start == 0
    assert chunks[1].token_start == 15
    assert chunks[0].token_end - chunks[1].token_start == 5
    assert all(chunk.token_count <= 20 for chunk in chunks)
    assert [encoding.encode(chunk.text) for chunk in chunks] == [
        token_ids[chunk.token_start : chunk.token_end] for chunk in chunks
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_chunk_pages_rejects_invalid_windows(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_pages(
            [PageText(page_number=1, text="content")],
            Path("demo.pdf"),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )


def test_chunks_never_cross_page_boundaries() -> None:
    chunks = chunk_pages(
        [PageText(1, "first page"), PageText(2, "second page")],
        Path("demo.pdf"),
        chunk_size=100,
        chunk_overlap=10,
    )

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert [chunk.chunk_index for chunk in chunks] == [0, 0]
