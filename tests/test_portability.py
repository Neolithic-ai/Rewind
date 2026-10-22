"""Installed-package artifact paths and eval subprocesses work outside the checkout."""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from rewind.evals import Evals, markdown, stats


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class RunsDirectory(unittest.TestCase):
    def read_runs_dir(self, cwd: Path, configured: str | None = None) -> Path:
        env = dict(os.environ)
        env.pop("REWIND_RUNS_DIR", None)
        if configured is not None:
            env["REWIND_RUNS_DIR"] = configured
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(PACKAGE_ROOT), env.get("PYTHONPATH")]))
        result = subprocess.run(
            [sys.executable, "-c", "from rewind.env import RUNS_DIR; print(RUNS_DIR)"],
            cwd=cwd, env=env, capture_output=True, text=True, check=True)
        return Path(result.stdout.strip())

    def test_default_uses_invocation_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            self.assertEqual(self.read_runs_dir(cwd), cwd / "runs")

    def test_relative_override_uses_invocation_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            self.assertEqual(self.read_runs_dir(cwd, "artifacts/experiment"),
                             cwd / "artifacts" / "experiment")

    def test_absolute_override_can_be_outside_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp) / "project"
            cwd.mkdir()
            output = Path(tmp) / "shared artifacts"
            self.assertEqual(self.read_runs_dir(cwd, str(output)), output)


class PortableEvals(unittest.TestCase):
    def test_subprocess_keeps_cwd_and_uses_parent_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            out = root / "output"
            (project / "input.txt").write_text("local task data")
            (project / "scenario.yaml").write_text(
                "name: portable\nservices: {}\nscorer: {service: default}\nagent: {}\n")
            # A tiny child runner exercises the real subprocess boundary without Docker.
            (project / "launch_probe.py").write_text(textwrap.dedent("""\
                import json
                import os
                from pathlib import Path
                assert Path("input.txt").read_text() == "local task data"
                output = Path(os.environ["REWIND_RUNS_DIR"]) / "probe"
                output.mkdir(parents=True)
                (output / "run.json").write_text(json.dumps({
                    "project": "probe", "kind": "run", "score": {"success": True},
                    "turns": 0, "usage": {"cost_usd": 0}, "checkpoints": []
                }))
                print("RUN_ID=probe")
                """))
            args = argparse.Namespace(scenario=str(project), model="mockllm/model",
                                      max_total_turns=None, budget_usd=None)
            evaluation = Evals(args, out / "eval", [])
            with contextlib.chdir(project), patch("rewind.evals.RUNS_DIR", out):
                result = evaluation.launch("launch_probe", [], "RUN_ID")
            self.assertTrue(result["success"])
            self.assertEqual(evaluation.errors, 0)
            self.assertTrue((out / "probe" / "run.json").is_file())
            self.assertTrue((out / "eval" / "runs" / "probe" / "run.json").is_file())

    def test_report_accepts_output_outside_checkout(self) -> None:
        result = {
            "scenario": "portable", "model": "mockllm/model", "full_runs": stats([]),
            "fork_points": [], "variants": [], "reach": {}, "branches": [],
            "parent_isolation": {},
            "totals": {"turns": 0, "cost_usd": 0, "wall_seconds": 0,
                       "stopped": None, "errors": 0},
            "verdict": {"ok": False, "reasons": ["no branches"], "failed_checks": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "shared results"
            report = markdown(result, out)
            self.assertIn(str(out / "results.json"), report)


if __name__ == "__main__":
    unittest.main()
