# ExCort

ExCort is a minimal PDF-only RAG chatbot built without LangChain or LlamaIndex.
It targets Python 3.12 and uses `uv` for reproducible package management.

## Setup

```bash
uv sync
cp .env.example .env
```

Place your licensed PDF at `data/raw/demo.pdf`, then set `JINA_API_KEY` and
`OPENROUTER_API_KEY` in `.env`. The PDF, API keys, generated chunks, and local
vector database are ignored by Git.

Build the local index:

```bash
uv run python scripts/phase1_check.py
uv run python scripts/phase2_check.py
```

## Run the application

Start the API and UI in separate terminals:

```bash
uv run uvicorn excort.api:app --reload
```

```bash
uv run streamlit run src/excort/frontend.py
```

Open `http://localhost:8501`. FastAPI documentation is available at
`http://localhost:8000/docs`.

## Quality checks

```bash
uv run python scripts/phase7_check.py
```

## Project layout

- `src/excort/`: application and RAG pipeline code
- `scripts/`: one observable check script per implementation phase
- `tests/`: automated tests
- `data/raw/demo.pdf`: user-supplied source document (ignored by Git)
- `data/processed/`: generated chunks (ignored by Git)
- `data/chroma/`: local Chroma database (ignored by Git)

## Current limitations

- Only text-based PDFs are supported; scanned files require out-of-scope OCR.
- The MVP indexes one fixed PDF and rebuilds its Chroma collection as a whole.
- Retrieval is dense-only and answers are generated without conversation memory.
- Hosted Jina and OpenRouter APIs require network access and may incur usage costs.

## Future work

- Add reranking with Jina Reranker.
- Add structured logging with Loguru.
- Add RAG evaluation with Ragas.
- Store chat history in SQLite.
- Package deployment with Docker.
- Add BM25 for hybrid search.
