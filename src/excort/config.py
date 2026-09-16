"""Centralized, environment-driven application configuration."""

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load configuration from environment variables and a local ``.env`` file.

    API keys stay optional during setup and ingestion so those phases can run offline.
    The clients that need a key will validate it at their own boundary.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openrouter_api_key: str | None = None
    jina_api_key: str | None = None
    openrouter_model: str = "deepseek/deepseek-v4-flash-0731"
    jina_embedding_model: str = "jina-embeddings-v5-omni-small"

    pdf_path: Path = Path("data/raw/demo.pdf")
    chroma_path: Path = Path("data/chroma")
    chroma_collection: str = "excort_documents"

    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=4, gt=0)
    backend_url: str = "http://127.0.0.1:8000"

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        """Prevent chunks that cannot advance through the token stream."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


settings = Settings()
