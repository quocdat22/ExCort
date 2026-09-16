"""Print observable evidence that the Phase 0 project setup is usable."""

import platform
from pathlib import Path

from excort.config import settings


def main() -> None:
    required_paths = [
        Path("pyproject.toml"),
        Path(".env.example"),
        Path("src/excort"),
        Path("data/raw/demo.pdf"),
        Path("scripts"),
        Path("tests"),
    ]

    print(f"Python: {platform.python_version()}")
    print("Required paths:")
    for path in required_paths:
        print(f"  [{'OK' if path.exists() else 'MISSING'}] {path}")

    print("Configuration:")
    print(f"  PDF: {settings.pdf_path}")
    print(f"  Chunking: size={settings.chunk_size}, overlap={settings.chunk_overlap}")
    print(f"  OpenRouter model: {settings.openrouter_model}")
    print(f"  Jina model: {settings.jina_embedding_model}")
    print("  API keys: hidden (and not required until their API phases)")

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Phase 0 check failed; missing: {', '.join(missing)}")
    print("Phase 0 check: PASS")


if __name__ == "__main__":
    main()
