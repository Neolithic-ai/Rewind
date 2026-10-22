"""Internal Docker integration check for the Inspect adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model

from inspect_rewind.doctor import doctor
from inspect_rewind.fork import fork, mock_outputs, snapshots


_COMPOSE = """services:
  default:
    image: debian:bookworm-slim
    command: [sleep, infinity]
    init: true
    working_dir: /root
    x-rewind:
      capture: [/root]
"""

_TASK = '''from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState
from inspect_ai.tool import bash
from inspect_ai.util import sandbox


@scorer(metrics=[accuracy()])
def count_matches():
    async def score(state: TaskState, target: Target) -> Score:
        result = await sandbox().exec(["bash", "-c", "cat /root/count.txt"])
        expected = "17" if (state.metadata or {}).get("rewind_branch") else "107"
        ok = result.success and result.stdout.strip() == expected
        return Score(value=CORRECT if ok else INCORRECT, explanation=result.stdout.strip())
    return score


@task
def rewind_smoke():
    return Task(
        dataset=[Sample(id="counter", input="Write 7 to /root/count.txt, then add 100 and submit the count.")],
        solver=react(tools=[bash(timeout=30)]),
        scorer=count_matches(),
        sandbox=("docker", str(Path(__file__).with_name("compose.yaml"))),
        message_limit=12,
    )
'''


def run_check(output_dir: Path) -> dict[str, Any]:
    """Run a real Docker rewind with two scripted models and no API key.

    Creates a unique run directory under ``output_dir`` containing its task,
    Compose config, snapshots, eval logs and ``report.json``. Returns a report
    with ``ok``, named boolean ``checks``, and artifact paths. Failures are
    recorded in the report, including failed prerequisite checks. Provider
    credentials are never read. Containers are managed by Inspect's cleanup.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="inspect-", dir=output_dir))
    task_file = run_dir / "task.py"
    task_file.write_text(_TASK)
    (run_dir / "compose.yaml").write_text(_COMPOSE)
    report: dict[str, Any] = {
        "ok": False,
        "run_dir": str(run_dir),
        "report_file": str(run_dir / "report.json"),
        "task_file": str(task_file),
        "snapshot_root": str(run_dir / "snapshots"),
        "sample_id": "counter",
        "step": 1,
        "checks": {},
    }
    previous = {key: os.environ.get(key)
                for key in ("INSPECT_REWIND_DIR", "INSPECT_REWIND_CAPTURE")}
    try:
        preflight = doctor()
        if not preflight["ok"]:
            report["prerequisites"] = preflight
            raise RuntimeError("Prerequisites are missing; run inspect-rewind doctor for fixes.")
        os.environ["INSPECT_REWIND_DIR"] = report["snapshot_root"]
        # Keep this check independent of capture overrides for a user's
        # own tasks, and restore the previous environment even after a failure.
        os.environ.pop("INSPECT_REWIND_CAPTURE", None)
        parent_name, branch_name = "mockllm/rewind-parent", "mockllm/rewind-branch"
        model = get_model(parent_name, custom_outputs=mock_outputs([
            {"bash": "printf '7\\n' > /root/count.txt"},
            {"bash": "printf '107\\n' > /root/count.txt; touch /root/parent-only.txt"},
            {"submit": "107"},
        ], parent_name))
        logs = inspect_eval(f"{task_file}@rewind_smoke", model=model, sandbox="rewind",
                            log_dir=str(run_dir / "logs"), display="none")
        parent = read_eval_log(logs[0].location, resolve_attachments=True)
        report["parent_log"] = parent.location
        checks = report["checks"]
        checks["parent_completed"] = parent.status == "success"
        if not checks["parent_completed"] or not parent.samples:
            raise RuntimeError(f"Parent evaluation did not complete: {parent.status}. See its eval log.")
        checks["parent_count_107"] = parent.samples[0].scores["count_matches"].value == "C"
        parent_steps = snapshots(parent.samples[0])
        checks["checkpoint_available"] = 1 in parent_steps
        if not checks["checkpoint_available"]:
            raise RuntimeError("The parent recorded no checkpoint for the first tool call.")
        report["checkpoint"] = parent_steps[1]
        digest = hashlib.sha256(Path(parent.location).read_bytes()).hexdigest()
        branch, branch_report = fork(
            parent.location, "counter", 1, model=branch_name,
            inject="Continue from 7 by adding 10 instead of 100; submit 17.",
            outputs=mock_outputs([
                {"bash": "test \"$(cat /root/count.txt)\" = 7 && "
                         "test ! -e /root/parent-only.txt && "
                         "printf '17\\n' > /root/count.txt"},
                {"submit": "17"},
            ], branch_name),
            log_dir=str(run_dir / "logs"),
        )
        report["branch_log"] = branch.location
        report["branch"] = branch_report
        checks.update({
            "branch_completed": branch.status == "success",
            "restore_verified": bool(branch_report.get("restore_ok")),
            "transcript_prefix_preserved": bool(branch_report.get("prefix_ok")),
            "instruction_injected": bool(branch_report.get("inject_ok")),
            "model_switched": parent.eval.model == parent_name and branch.eval.model == branch_name,
            "parent_log_unchanged": hashlib.sha256(Path(parent.location).read_bytes()).hexdigest() == digest,
            # This only reaches 17 after verifying both the captured value and
            # the absence of a file created after the checkpoint.
            "branch_restored_7_then_reached_17": bool(
                branch.samples and branch.samples[0].scores
                and branch.samples[0].scores.get("count_matches")
                and branch.samples[0].scores["count_matches"].value == "C"
            ),
        })
        if branch.samples:
            report["branch_checkpoints"] = snapshots(branch.samples[0])
        report["ok"] = all(checks.values())
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        Path(report["report_file"]).write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_check(args.output_dir)
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["ok"] else 1)
