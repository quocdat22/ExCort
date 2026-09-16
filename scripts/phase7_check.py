"""Run all offline tests and code-quality gates for the MVP."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_check(label: str, command: list[str]) -> None:
    """Run one gate, stream its result, and stop on failure."""
    print(f"\n=== {label} ===")
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{label}: FAIL (exit code {result.returncode})")
    print(f"{label}: PASS")


def main() -> None:
    run_check("pytest", [sys.executable, "-m", "pytest", "-q"])
    run_check("Ruff lint", [sys.executable, "-m", "ruff", "check", "."])
    run_check(
        "Ruff format",
        [sys.executable, "-m", "ruff", "format", "--check", "."],
    )
    run_check("Git whitespace", ["git", "diff", "--check"])
    print("\nPhase 7 check: PASS")


if __name__ == "__main__":
    main()
