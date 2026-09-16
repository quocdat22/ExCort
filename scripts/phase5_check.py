"""Start Uvicorn, call the real HTTP endpoints, and print their responses."""

import json
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx

from excort.config import settings

QUESTION = "What is MLOps, and how is it related to ML systems design?"
STARTUP_TIMEOUT_SECONDS = 20


def wait_until_ready(client: httpx.Client, process: subprocess.Popen[str]) -> dict:
    """Poll health until Uvicorn is ready or exits unexpectedly."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"Uvicorn exited early:\n{output}")
        try:
            response = client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as error:
            last_error = error
            time.sleep(0.2)
    raise TimeoutError(f"API did not become ready: {last_error}")


def main() -> None:
    parsed_url = urlparse(settings.backend_url)
    if parsed_url.scheme != "http" or not parsed_url.hostname or not parsed_url.port:
        raise SystemExit("BACKEND_URL must be an explicit local HTTP URL with a port")

    command = [
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
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with httpx.Client(base_url=settings.backend_url, timeout=180.0) as client:
            health_response = wait_until_ready(client, process)
            chat_response = client.post(
                "/chat",
                json={"question": QUESTION, "top_k": settings.top_k},
            )
            chat_response.raise_for_status()
            chat_payload = chat_response.json()

        print("GET /health")
        print(json.dumps(health_response, indent=2, ensure_ascii=False))
        print("\nPOST /chat")
        print(json.dumps(chat_payload, indent=2, ensure_ascii=False))

        if health_response["record_count"] <= 0:
            raise SystemExit("Phase 5 check failed: vector collection is empty")
        if not chat_payload["answer"] or not chat_payload["sources"]:
            raise SystemExit("Phase 5 check failed: incomplete chat response")
        print("Phase 5 check: PASS")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
