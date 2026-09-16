"""Small HTTP client for Jina's hosted embedding API."""

from collections.abc import Sequence
from typing import Any, Literal

import httpx

JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
EmbeddingTask = Literal["retrieval.passage", "retrieval.query"]


class JinaEmbeddingClient:
    """Embed passages and queries without a RAG framework dependency."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        batch_size: int = 16,
        timeout: float = 60.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("JINA_API_KEY is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed index content with Jina's passage-specific retrieval adapter."""
        return self._embed_batched(texts, task="retrieval.passage")

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query in the same vector space as document passages."""
        if not query.strip():
            raise ValueError("query must not be empty")
        return self._request_embeddings([query], task="retrieval.query")[0]

    def _embed_batched(
        self, texts: Sequence[str], task: EmbeddingTask
    ) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("embedding inputs must not be empty")

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            embeddings.extend(self._request_embeddings(batch, task=task))
        return embeddings

    def _request_embeddings(
        self, texts: Sequence[str], task: EmbeddingTask
    ) -> list[list[float]]:
        # Jina v5 uses different retrieval adapters for indexed passages and queries.
        # Explicit normalization makes cosine behavior predictable in Chroma.
        payload = {
            "model": self.model,
            "task": task,
            "normalized": True,
            "embedding_type": "float",
            "input": [{"text": text} for text in texts],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(JINA_EMBEDDINGS_URL, headers=headers, json=payload)
            response.raise_for_status()
        return self._parse_response(response.json(), expected_count=len(texts))

    @staticmethod
    def _parse_response(payload: Any, expected_count: int) -> list[list[float]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("Jina response is missing the data list")

        ordered_items = sorted(payload["data"], key=lambda item: item.get("index", -1))
        embeddings = [item.get("embedding") for item in ordered_items]
        if len(embeddings) != expected_count:
            raise ValueError(
                f"Jina returned {len(embeddings)} embeddings; expected {expected_count}"
            )
        if any(
            not isinstance(vector, list)
            or not vector
            or not all(isinstance(value, int | float) for value in vector)
            for vector in embeddings
        ):
            raise ValueError("Jina returned an invalid embedding vector")
        dimensions = {len(vector) for vector in embeddings}
        if len(dimensions) != 1:
            raise ValueError("Jina returned inconsistent embedding dimensions")
        return embeddings
