"""Exercise the Streamlit UI against a live FastAPI backend."""

import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from streamlit.testing.v1 import AppTest

from excort.config import settings

QUESTION = "What is MLOps?"


def start_backend() -> subprocess.Popen[str]:
    """Start the backend used by the Streamlit integration check."""
    parsed_url = urlparse(settings.backend_url)
    if parsed_url.scheme != "http" or not parsed_url.hostname or not parsed_url.port:
        raise ValueError("BACKEND_URL must be an explicit HTTP URL with a port")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "excort.api:app",
            "--host",
            parsed_url.hostname,
            "--port",
            str(parsed_url.port),
            "--log-level",
            "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for_backend(process: subprocess.Popen[str]) -> None:
    """Wait until FastAPI is ready before Streamlit calls it."""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"Uvicorn exited early:\n{output}")
        try:
            response = httpx.get(f"{settings.backend_url}/health", timeout=1.0)
            response.raise_for_status()
            return
        except httpx.HTTPError:
            time.sleep(0.2)
    raise TimeoutError("FastAPI did not become ready")


def main() -> None:
    process = start_backend()
    try:
        wait_for_backend(process)
        frontend_path = Path(__file__).resolve().parents[1] / "src/excort/frontend.py"
        app = AppTest.from_file(frontend_path, default_timeout=180)
        app.run()
        if app.exception:
            raise RuntimeError(f"Initial Streamlit render failed: {app.exception}")
        print(f"Page title: {app.title[0].value}")
        print(f"Chat input: {app.chat_input[0].placeholder}")
        print(f"Sidebar status: {app.sidebar.success[0].value}")

        app.chat_input[0].set_value(QUESTION).run()
        if app.exception:
            raise RuntimeError(f"Chat interaction failed: {app.exception}")
        messages = app.session_state["messages"]
        if len(messages) != 2 or messages[-1]["role"] != "assistant":
            raise RuntimeError("Streamlit did not store the expected conversation")
        if not messages[-1]["content"] or not messages[-1]["sources"]:
            raise RuntimeError("Assistant message is missing an answer or sources")

        print(f"Stored messages: {len(messages)}")
        print(f"Assistant answer: {messages[-1]['content']}")
        print(f"Displayed sources: {len(messages[-1]['sources'])}")
        print("Phase 6 check: PASS")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
