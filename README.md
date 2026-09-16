# ExCort

ExCort is a minimal PDF-only RAG chatbot built without LangChain or LlamaIndex.
It targets Python 3.12 and uses `uv` for reproducible package management.

## Setup

```bash
uv sync
cp .env.example .env
uv run python scripts/phase0_check.py
```

API keys may remain empty through ingestion. Before the embedding and generation
phases, set `JINA_API_KEY` and `OPENROUTER_API_KEY` in `.env`.

## Project layout

- `src/excort/`: application and RAG pipeline code
- `scripts/`: one observable check script per implementation phase
- `tests/`: automated tests
- `data/raw/demo.pdf`: source document
- `data/processed/`: generated chunks (ignored by Git)
- `data/chroma/`: local Chroma database (ignored by Git)

## Future work

- Add reranking with Jina Reranker.
- Add structured logging with Loguru.
- Add RAG evaluation with Ragas.
- Store chat history in SQLite.
- Package deployment with Docker.
- Add BM25 for hybrid search.

