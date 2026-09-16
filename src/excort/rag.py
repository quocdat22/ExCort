"""End-to-end retrieval-augmented generation orchestration."""

from dataclasses import dataclass

from excort.generation import FinalPrompt, OpenRouterClient, build_prompt
from excort.retrieval import DenseRetriever, RetrievedChunk


@dataclass(frozen=True)
class RagResponse:
    """Answer plus the evidence and prompt used to produce it."""

    answer: str
    sources: list[RetrievedChunk]
    prompt: FinalPrompt


class RagChain:
    """Compose dense retrieval, prompt building, and answer generation."""

    def __init__(self, retriever: DenseRetriever, llm: OpenRouterClient) -> None:
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, top_k: int) -> RagResponse:
        """Answer one question from the highest-scoring document chunks."""
        sources = self.retriever.search(question, top_k=top_k)
        prompt = build_prompt(question, sources)
        answer = self.llm.generate(prompt)
        return RagResponse(answer=answer, sources=sources, prompt=prompt)
