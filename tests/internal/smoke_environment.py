"""Internal Docker integration check against the installed standalone package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from rewind.workspace import Checkpoint, Environment


_COMPOSE = """services:
  default:
    image: debian:bookworm-slim
    command: [sleep, infinity]
    init: true
    working_dir: /root
"""


def _exec(environment: Environment, command: list[str]) -> str:
    code, output = environment.exec("default", command)
    if code:
        raise RuntimeError(f"smoke command failed in {environment.project} ({code}): {output}")
    return output.strip()


def _removed(project: str) -> bool:
    for args in (["docker", "ps", "-aq"], ["docker", "volume", "ls", "-q"],
                 ["docker", "network", "ls", "-q"]):
        result = subprocess.run([*args, "--filter", f"label=com.docker.compose.project={project}"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode or result.stdout.strip():
            return False
    return True


def run_check(output_dir: Path) -> dict[str, Any]:
    """Exercise parent/branch isolation and JSON state with scripted commands.

    This is an environment smoke test, not a real agent run. Artifacts remain
    in a unique directory under ``output_dir``; both environments are removed
    before returning. A failed check returns ``ok=False`` and writes a report.
    """
    from rewind.cli import _doctor

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="environment-", dir=output_dir))
    compose = run_dir / "compose.yaml"
    compose.write_text(_COMPOSE)
    report: dict[str, Any] = {
        "ok": False,
        "mode": "scripted-environment-check",
        "run_dir": str(run_dir),
        "report_file": str(run_dir / "report.json"),
        "compose_file": str(compose),
        "checks": {},
    }
    environments: list[Environment] = []
    try:
        prerequisites = _doctor()
        if not prerequisites["ok"]:
            report["prerequisites"] = prerequisites
            raise RuntimeError("Docker prerequisites are missing; run rewind doctor for fixes.")
        parent = Environment.from_compose(compose, capture={"default": ["/root"]})
        environments.append(parent)
        parent.start()
        report["parent_project"] = parent.project
        _exec(parent, ["sh", "-c", "printf '7\\n' > /root/count.txt"])
        report["values"] = {"checkpoint": int(_exec(parent, ["cat", "/root/count.txt"]))}
        metadata = {"step": 1, "source": "integration-test", "model": "scripted-parent"}
        messages = [{"role": "tool", "name": "exec", "content": "Counter set to 7."}]
        checkpoint = parent.checkpoint(run_dir / "checkpoint", metadata=metadata, messages=messages)
        report["checkpoint"] = str(checkpoint.path)
        saved = Checkpoint.load(checkpoint.path)
        checks = report["checks"]
        checks["metadata_roundtrip"] = saved.metadata == metadata
        checks["messages_roundtrip"] = saved.messages == messages
        before_hashes = saved.verify()
        _exec(parent, ["sh", "-c", 'value=$(cat /root/count.txt); printf "%s\\n" "$((value + 100))" '
                     '> /root/count.txt; touch /root/parent-only.txt'])
        report["values"]["parent"] = int(_exec(parent, ["cat", "/root/count.txt"]))
        checks["parent_reached_107"] = report["values"]["parent"] == 107
        branch = saved.branch()
        environments.append(branch)
        report["branch_project"] = branch.project
        report["restore"] = branch.restore_report
        checks["isolated_projects"] = parent.project != branch.project
        checks["restore_verified"] = bool(branch.restore_report and branch.restore_report.get("ok"))
        report["values"]["branch_restored"] = int(_exec(branch, ["cat", "/root/count.txt"]))
        checks["branch_restored_7"] = report["values"]["branch_restored"] == 7
        absent, _ = branch.exec("default", ["test", "!", "-e", "/root/parent-only.txt"])
        checks["later_parent_file_absent"] = absent == 0
        _exec(branch, ["sh", "-c", 'value=$(cat /root/count.txt); printf "%s\\n" "$((value + 10))" '
                     '> /root/count.txt'])
        report["values"]["branch"] = int(_exec(branch, ["cat", "/root/count.txt"]))
        checks["branch_reached_17"] = report["values"]["branch"] == 17
        report["values"]["parent_after_branch"] = int(_exec(parent, ["cat", "/root/count.txt"]))
        checks["parent_remained_107"] = report["values"]["parent_after_branch"] == 107
        checks["checkpoint_unchanged"] = saved.verify() == before_hashes
        report["capabilities"] = saved.manifest["capabilities"]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = []
        for environment in reversed(environments):
            try:
                environment.down()
                cleanup.append({"project": environment.project, "ok": _removed(environment.project)})
            except Exception as exc:
                cleanup.append({"project": environment.project, "ok": False, "error": str(exc)})
        report["cleanup"] = cleanup
        if cleanup:
            report["checks"]["environments_removed"] = all(item["ok"] for item in cleanup)
        report["ok"] = bool(report["checks"]) and all(report["checks"].values()) and "error" not in report
        Path(report["report_file"]).write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_check(args.output_dir)
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["ok"] else 1)
