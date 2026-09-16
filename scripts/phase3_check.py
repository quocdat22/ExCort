"""Run dense retrieval and display ranked chunks with similarity scores."""

import argparse

from excort.config import settings
from excort.embeddings import JinaEmbeddingClient
from excort.retrieval import DenseRetriever

DEFAULT_QUERY = "What is the relationship between MLOps and ML systems design?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Question to embed and retrieve against the local Chroma collection.",
    )
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not settings.jina_api_key:
        raise SystemExit("JINA_API_KEY is missing from .env")

    embedder = JinaEmbeddingClient(
        api_key=settings.jina_api_key,
        model=settings.jina_embedding_model,
    )
    retriever = DenseRetriever(
        embedder=embedder,
        persist_path=settings.chroma_path,
        collection_name=settings.chroma_collection,
    )
    results = retriever.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Dense top-k: {len(results)}")
    for rank, result in enumerate(results, start=1):
        page = result.metadata["page_number"]
        sample = result.text[:500].replace("\n", " ")
        print(
            f"\n#{rank} id={result.id} page={page} "
            f"similarity={result.similarity:.4f}\n{sample}"
        )

    if len(results) != min(args.top_k, retriever.collection.count()):
        raise SystemExit("Phase 3 check failed: unexpected result count")
    if any(
        left.similarity < right.similarity
        for left, right in zip(results, results[1:], strict=False)
    ):
        raise SystemExit("Phase 3 check failed: results are not ranked")
    print("\nPhase 3 check: PASS")


if __name__ == "__main__":
    main()
