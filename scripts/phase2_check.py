"""Embed all chunks, persist them to Chroma, and run one raw query."""

import json
from pathlib import Path

from excort.config import settings
from excort.embeddings import JinaEmbeddingClient
from excort.ingestion import Chunk
from excort.vector_store import rebuild_collection

CHUNKS_PATH = Path("data/processed/chunks.json")
SAMPLE_QUERY = "What is MLOps and how does it relate to ML systems design?"


def load_chunks(path: Path) -> list[Chunk]:
    """Load Phase 1 output and validate every item through the Chunk constructor."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Chunk file not found: {path}. Run scripts/phase1_check.py first."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**row) for row in rows]


def main() -> None:
    if not settings.jina_api_key:
        raise SystemExit("JINA_API_KEY is missing from .env")

    chunks = load_chunks(CHUNKS_PATH)
    embedder = JinaEmbeddingClient(
        api_key=settings.jina_api_key,
        model=settings.jina_embedding_model,
    )
    print(f"Embedding {len(chunks)} chunks with {settings.jina_embedding_model}...")
    embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
    collection = rebuild_collection(
        persist_path=settings.chroma_path,
        collection_name=settings.chroma_collection,
        chunks=chunks,
        embeddings=embeddings,
    )

    query_embedding = embedder.embed_query(SAMPLE_QUERY)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )

    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Records in Chroma: {collection.count()}")
    print(f"Raw query: {SAMPLE_QUERY}")
    print(f"Top result ID: {result['ids'][0][0]}")
    print(f"Cosine distance: {result['distances'][0][0]:.4f}")
    print(f"Metadata: {result['metadatas'][0][0]}")
    sample = result["documents"][0][0][:500].replace("\n", " ")
    print(f"Text sample: {sample}")

    if collection.count() != len(chunks):
        raise SystemExit("Phase 2 check failed: Chroma record count mismatch")
    print("Phase 2 check: PASS")


if __name__ == "__main__":
    main()
