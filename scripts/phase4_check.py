"""Run the complete RAG chain and print its final prompt and answer."""

import argparse

from excort.config import settings
from excort.embeddings import JinaEmbeddingClient
from excort.generation import OpenRouterClient
from excort.rag import RagChain
from excort.retrieval import DenseRetriever

DEFAULT_QUESTION = "What is MLOps, and how is it related to ML systems design?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not settings.jina_api_key:
        raise SystemExit("JINA_API_KEY is missing from .env")
    if not settings.openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing from .env")

    embedder = JinaEmbeddingClient(
        api_key=settings.jina_api_key,
        model=settings.jina_embedding_model,
    )
    retriever = DenseRetriever(
        embedder=embedder,
        persist_path=settings.chroma_path,
        collection_name=settings.chroma_collection,
    )
    llm = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
    )
    response = RagChain(retriever=retriever, llm=llm).answer(
        args.question, top_k=args.top_k
    )

    print("=== FINAL PROMPT: SYSTEM ===")
    print(response.prompt.system)
    print("\n=== FINAL PROMPT: USER ===")
    print(response.prompt.user)
    print("\n=== ANSWER ===")
    print(response.answer)
    print("\n=== RETRIEVED SOURCES ===")
    for source in response.sources:
        print(
            f"- {source.id}, page {source.metadata['page_number']}, "
            f"similarity={source.similarity:.4f}"
        )

    if not response.answer or not response.sources:
        raise SystemExit("Phase 4 check failed: answer or sources are empty")
    print("Phase 4 check: PASS")


if __name__ == "__main__":
    main()
