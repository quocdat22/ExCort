"""Tests for dense result conversion and ranking semantics."""

from pathlib import Path

import pytest

import excort.retrieval as retrieval_module
from excort.retrieval import DenseRetriever


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


class FakeCollection:
    def count(self) -> int:
        return 2

    def query(self, **kwargs: object) -> dict:
        assert kwargs["n_results"] == 2
        return {
            "ids": [["first", "second"]],
            "documents": [["most relevant", "less relevant"]],
            "metadatas": [[{"page_number": 1}, {"page_number": 2}]],
            "distances": [[0.1, 0.4]],
        }


def test_dense_retriever_converts_cosine_distance_to_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retrieval_module, "get_collection", lambda *_args: FakeCollection()
    )
    retriever = DenseRetriever(FakeEmbedder(), Path("unused"), "collection")

    results = retriever.search("question", top_k=5)

    assert [result.id for result in results] == ["first", "second"]
    assert [result.similarity for result in results] == pytest.approx([0.9, 0.6])


@pytest.mark.parametrize(("query", "top_k"), [("", 1), ("question", 0)])
def test_dense_retriever_validates_search_input(
    monkeypatch: pytest.MonkeyPatch, query: str, top_k: int
) -> None:
    monkeypatch.setattr(
        retrieval_module, "get_collection", lambda *_args: FakeCollection()
    )
    retriever = DenseRetriever(FakeEmbedder(), Path("unused"), "collection")
    with pytest.raises(ValueError):
        retriever.search(query, top_k)
