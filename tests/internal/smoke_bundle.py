"""Check installed-wheel bundle portability outside the source checkout."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import rewind
from rewind import Checkpoint, Environment


def cli(*args: str) -> dict[str, Any]:
    result = subprocess.run([sys.executable, "-m", "rewind.cli", *args],
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"rewind {args[0]} failed: {result.stderr or result.stdout}")
    return json.loads(result.stdout)


def execute(environment: Environment, command: str) -> str:
    code, output = environment.exec("default", command)
    if code:
        raise RuntimeError(f"container command failed ({code}): {output}")
    return output


def removed(project: str) -> bool:
    for args in (["docker", "ps", "-aq"], ["docker", "volume", "ls", "-q"],
                 ["docker", "network", "ls", "-q"]):
        result = subprocess.run([*args, "--filter", f"label=com.docker.compose.project={project}"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode or result.stdout.strip():
            return False
    return True


def run_check(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="bundle-", dir=output_dir))
    report: dict[str, Any] = {"ok": False, "run_dir": str(root), "checks": {}}
    checks = report["checks"]
    environments: list[Environment] = []
    try:
        report["package_file"] = rewind.__file__
        checks["uses_installed_wheel"] = Path(rewind.__file__).is_relative_to(Path(sys.prefix))
        checks["inspect_not_installed"] = importlib.util.find_spec("inspect_ai") is None
        result = subprocess.run([sys.executable, "-m", "rewind.cli", "--version"],
                                capture_output=True, text=True, timeout=10)
        checks["version_matches_distribution"] = (
            result.returncode == 0 and result.stdout.strip() ==
            f"rewind {importlib.metadata.version('rewind-env')}"
        )
        source = root / "original"
        inputs = source / "input$dir"
        inputs.mkdir(parents=True)
        (inputs / "empty").mkdir()
        (inputs / "setting.txt").write_text("saved input")
        compose = source / "compose.yaml"
        compose.write_text("""services:
  default:
    image: debian:bookworm-slim
    command: [sleep, infinity]
    environment:
      LITERAL: 'a$$dollar'
    volumes:
      - state:/data
      - ./input$$dir:/inputs:ro
volumes:
  state:
""")
        parent = Environment.from_compose(compose, capture={"default": ["/data"]})
        environments.append(parent)
        parent.start()
        execute(parent, "printf 7 > /data/count")
        messages = [{"role": "user", "content": "Continue from seven."}]
        checkpoint = parent.checkpoint(source / "checkpoint", messages=messages,
                                       metadata={"step": 1, "model": "internal-script"})
        api_archive = checkpoint.export(root / "api.rewind")
        cli_archive = cli("export", str(checkpoint.path), "--output", str(root / "cli.rewind"))["bundle"]
        execute(parent, "printf 99 > /data/count")
        parent.down()
        shutil.rmtree(source)
        checks["original_inputs_and_checkpoint_removed"] = not source.exists()

        imported = Checkpoint.import_bundle(api_archive, root / "api-import")
        checks["api_preserves_messages_and_metadata"] = (
            imported.messages == messages and imported.metadata == checkpoint.metadata
        )
        api_branch = imported.branch()
        environments.append(api_branch)
        checks["api_restores_selected_state"] = execute(api_branch, "cat /data/count") == "7"
        checks["api_restore_verified"] = bool(api_branch.restore_report["ok"])
        checks["readonly_input_survives_removal"] = execute(api_branch, "cat /inputs/setting.txt") == "saved input"
        checks["empty_input_directory_preserved"] = api_branch.exec("default", ["test", "-d", "/inputs/empty"])[0] == 0
        checks["literal_dollar_preserved"] = execute(api_branch, 'printf %s "$LITERAL"') == "a$dollar"

        cli_import = cli("import", cli_archive, "--output", str(root / "cli-import"))
        cli_checkpoint = Checkpoint.load(cli_import["checkpoint"])
        checks["cli_preserves_messages_and_metadata"] = (
            cli_checkpoint.messages == messages and cli_checkpoint.metadata == checkpoint.metadata
        )
        state = root / "branch.json"
        cli_branch = cli("branch", str(cli_checkpoint.path), "--state", str(state))
        branch = Environment.load(state)
        environments.append(branch)
        checks["cli_restore_verified"] = bool(cli_branch["restore"]["ok"])
        checks["cli_restores_selected_state"] = cli("exec", str(state), "--", "cat", "/data/count")["output"] == "7"
        cli("exec", str(state), "--", "sh", "-c", "printf 17 > /data/count")
        checks["imported_branches_are_independent"] = execute(api_branch, "cat /data/count") == "7"
        cli("down", str(state))
        environments[-1] = Environment.load(state)
        checks["cli_cleanup_persisted"] = not environments[-1].started
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = []
        for environment in reversed(environments):
            try:
                environment.down()
                cleanup.append({"project": environment.project, "ok": removed(environment.project)})
            except Exception as exc:
                cleanup.append({"project": environment.project, "ok": False, "error": str(exc)})
        report["cleanup"] = cleanup
        checks["all_projects_removed"] = bool(cleanup) and all(item["ok"] for item in cleanup)
        report["ok"] = bool(checks) and all(checks.values()) and "error" not in report
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = run_check(parser.parse_args().output_dir)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)
