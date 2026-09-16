"""Offline tests for Jina and OpenRouter HTTP contracts."""

import json

import httpx

from excort.embeddings import JinaEmbeddingClient
from excort.generation import FinalPrompt, OpenRouterClient


def test_jina_client_batches_documents_and_sets_passage_task() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        data = [
            {"index": index, "embedding": [float(index), 1.0]}
            for index, _item in enumerate(payload["input"])
        ]
        return httpx.Response(200, json={"data": data})

    client = JinaEmbeddingClient(
        "test-key",
        "test-model",
        batch_size=2,
        transport=httpx.MockTransport(handler),
    )
    vectors = client.embed_documents(["one", "two", "three"])

    assert len(requests) == 2
    assert all(request["task"] == "retrieval.passage" for request in requests)
    assert all(request["normalized"] is True for request in requests)
    assert vectors == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]


def test_jina_client_uses_query_task() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["task"] == "retrieval.query"
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5]}]})

    client = JinaEmbeddingClient(
        "test-key", "test-model", transport=httpx.MockTransport(handler)
    )
    assert client.embed_query("question") == [0.5]


def test_openrouter_client_sends_messages_and_parses_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek/test"
        assert [message["role"] for message in payload["messages"]] == [
            "system",
            "user",
        ]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": " Grounded answer "}}]},
        )

    client = OpenRouterClient(
        "test-key", "deepseek/test", transport=httpx.MockTransport(handler)
    )
    answer = client.generate(FinalPrompt(system="rules", user="context"))
    assert answer == "Grounded answer"


def test_clients_reject_empty_inputs() -> None:
    embedder = JinaEmbeddingClient("test-key", "test-model")
    assert embedder.embed_documents([]) == []
    try:
        embedder.embed_query(" ")
    except ValueError as error:
        assert "query" in str(error)
    else:
        raise AssertionError("empty query should fail")
