# Repository Guidelines

## Project Structure & Module Organization

Application code lives in `src/excort/`. Keep pipeline stages separate: PDF parsing
and chunking in `ingestion.py`, API clients in `embeddings.py` and `generation.py`,
vector persistence in `vector_store.py`, retrieval in `retrieval.py`, and orchestration
in `rag.py`. FastAPI endpoints and Pydantic contracts live in `api.py` and
`schemas.py`; `frontend.py` contains the Streamlit UI.

Tests belong in `tests/` and follow the source module names. `scripts/phase0_check.py`
through `phase7_check.py` are observable integration and quality checks. Runtime data
is under `data/`; `data/raw/demo.pdf`, processed chunks, Chroma files, and `.env` are
local-only and must remain untracked.

## Build, Test, and Development Commands

- `uv sync`: create the Python 3.12 environment and install locked dependencies.
- `uv run python scripts/phase1_check.py`: parse and chunk the local PDF.
- `uv run python scripts/phase2_check.py`: embed chunks and rebuild ChromaDB.
- `uv run uvicorn excort.api:app --reload`: run FastAPI on port 8000.
- `uv run streamlit run src/excort/frontend.py`: run the UI on port 8501.
- `uv run python scripts/phase7_check.py`: run all tests and quality gates.
- `uv run pytest -q`: run the offline unit/API suite directly.

## Coding Style & Naming Conventions

Use four-space indentation, Python 3.12 type hints, short docstrings, and comments
that explain important design choices rather than restating code. Ruff enforces an
88-character line length plus `E`, `F`, `I`, `UP`, and `B` rules. Run
`uv run ruff format .` and `uv run ruff check .` before committing. Use
`snake_case` for modules/functions/variables and `PascalCase` for classes and
Pydantic models.

## Testing Guidelines

Use pytest. Name files `test_<module>.py` and tests `test_<behavior>`. Tests must be
deterministic and offline: inject `httpx.MockTransport` for Jina/OpenRouter and use
FastAPI dependency overrides for RAG calls. Add tests for validation failures and
edge cases alongside the happy path. There is no numeric coverage gate; every
changed core behavior should have direct assertions.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit style: `feat:`, `test:`, `docs:`, and
`chore:` followed by a short imperative summary. Keep commits focused by phase or
concern. Pull requests should explain behavior changes, list verification commands,
link relevant issues, and include screenshots for Streamlit UI changes. Never include
API keys, generated databases, processed chunks, or copyrighted PDFs.

## Security & Configuration

Copy `.env.example` to `.env` and keep secrets local. Preserve the required model
defaults unless the project specification changes. Avoid logging authorization
headers, full API responses containing sensitive data, or local document contents.
