"""Dense-only retrieval over the persisted ExCort Chroma collection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from excort.embeddings import JinaEmbeddingClient
from excort.vector_store import get_collection


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked search result with an intuitive cosine similarity score."""

    id: str
    text: str
    metadata: dict[str, Any]
    similarity: float


class DenseRetriever:
    """Embed a query and search Chroma without keyword or reranking stages."""

    def __init__(
        self,
        embedder: JinaEmbeddingClient,
        persist_path: Path,
        collection_name: str,
    ) -> None:
        self.embedder = embedder
        self.collection = get_collection(persist_path, collection_name)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the most similar chunks in descending relevance order."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        record_count = self.collection.count()
        if record_count == 0:
            raise ValueError("Chroma collection is empty; run Phase 2 first")

        query_embedding = self.embedder.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, record_count),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []
        if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
            raise ValueError("Chroma returned incomplete query results")

        # For a cosine-distance collection, similarity = 1 - distance. Exposing
        # similarity makes a larger score consistently mean a better match.
        return [
            RetrievedChunk(
                id=chunk_id,
                text=document,
                metadata=dict(metadata),
                similarity=1.0 - float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]
