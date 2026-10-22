"""Read-only checks for the local Inspect and Docker installation."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
from typing import Any


def _command(args: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return False, "Command did not respond within 10 seconds."
    except OSError as exc:
        return False, str(exc)
    output = (result.stdout if result.returncode == 0 else result.stderr).strip()
    return result.returncode == 0, output[:500] or f"Exit status {result.returncode}."


def doctor() -> dict[str, Any]:
    """Check prerequisites without creating containers or contacting a provider.

    Return ``{"ok": bool, "checks": [{"name", "ok", "detail", "fix"}]}``.
    A successful check has an empty ``fix``. Each subprocess has a 10s timeout.
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, fix: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "fix": "" if ok else fix})

    add("python", sys.version_info >= (3, 12), sys.version.split()[0],
        "Install Python 3.12 or later and reinstall inspect-rewind in that environment.")
    try:
        version = importlib.metadata.version("inspect-ai")
        # Check the private interfaces used by the adapter as well as the
        # installed release metadata.
        from inspect_ai._eval.loader import load_tasks
        from inspect_ai.log._samples import sample_active
        from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment

        if not all(callable(symbol) for symbol in (load_tasks, sample_active, DockerSandboxEnvironment)):
            raise ImportError("Required Inspect interfaces are unavailable.")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        add("inspect", False, str(exc),
            "Install the Inspect adapter in this environment: pip install 'rewind-env[inspect]' "
            "(or pip install '.[inspect]' from a checkout).")
    else:
        add("inspect", True, version, "")

    try:
        entries = importlib.metadata.entry_points(group="inspect_ai")
        entry = next((ep for ep in entries
                      if ep.name == "inspect_rewind" and ep.value == "inspect_rewind._registry"), None)
        if entry is None:
            raise ImportError("The inspect_rewind entry point is not installed.")
        module = entry.load()
        if not hasattr(module, "RewindSandboxEnvironment"):
            raise ImportError("The installed entry point does not expose the rewind sandbox.")
    except Exception as exc:
        add("plugin", False, str(exc),
            "Install the adapter in the same Python environment as Inspect "
            "(pip install '.[inspect]' from a checkout).")
    else:
        add("plugin", True, "Inspect can load the rewind sandbox entry point.", "")

    docker = shutil.which("docker")
    add("docker_cli", docker is not None, docker or "docker is not on PATH.",
        "Install Docker Desktop or Docker Engine with the Docker Compose plugin.")
    if docker:
        ok, detail = _command([docker, "info", "--format", "{{.ServerVersion}}"])
        add("docker_daemon", ok, detail,
            "Start Docker and check that this user can run docker info in the selected Docker context.")
        ok, detail = _command([docker, "compose", "version", "--short"])
        add("docker_compose", ok, detail,
            "Install the Docker Compose v2 plugin; docker compose version must succeed.")
    else:
        for name in ("docker_daemon", "docker_compose"):
            add(name, False, "Not checked because docker is not on PATH.",
                "Install Docker, then run inspect-rewind doctor again.")
    return {"ok": all(check["ok"] for check in checks), "checks": checks}
