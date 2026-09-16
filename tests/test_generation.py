"""Tests for grounded prompt construction."""

import pytest

from excort.generation import build_prompt
from excort.retrieval import RetrievedChunk


def test_build_prompt_labels_context_and_question() -> None:
    chunk = RetrievedChunk(
        id="demo-p0002-c0000",
        text="MLOps brings ML into production.",
        metadata={"page_number": 2},
        similarity=0.9,
    )
    prompt = build_prompt("What is MLOps?", [chunk])

    assert 'page="2"' in prompt.user
    assert 'chunk_id="demo-p0002-c0000"' in prompt.user
    assert "<question>\nWhat is MLOps?\n</question>" in prompt.user
    assert "using only the supplied context" in prompt.system


@pytest.mark.parametrize(("question", "chunks"), [("", [object()]), ("question", [])])
def test_build_prompt_requires_question_and_context(
    question: str, chunks: list
) -> None:
    with pytest.raises(ValueError):
        build_prompt(question, chunks)
