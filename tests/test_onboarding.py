"""Prerequisite failures and internal integration-check safeguards."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inspect_rewind.cli import main
from inspect_rewind.doctor import doctor
from tests.internal.smoke_inspect import run_check


class DoctorTests(unittest.TestCase):
    def test_help_exposes_only_setup_and_branching_commands(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as result:
            main(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("{doctor,checkpoints,export,import,fork}", output.getvalue())
        self.assertNotIn("demo", output.getvalue())

    def test_fork_help_focuses_on_continuation_options(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as result:
            main(["fork", "--help"])
        self.assertEqual(result.exception.code, 0)
        for option in ("--step", "--model", "--inject"):
            self.assertIn(option, output.getvalue())
        self.assertNotIn("--mock-outputs", output.getvalue())

    def test_missing_docker_is_actionable_and_runs_no_commands(self) -> None:
        with patch("inspect_rewind.doctor.shutil.which", return_value=None), \
             patch("inspect_rewind.doctor.subprocess.run") as run:
            report = doctor()
        self.assertFalse(report["ok"])
        checks = {check["name"]: check for check in report["checks"]}
        self.assertFalse(checks["docker_cli"]["ok"])
        self.assertIn("Install Docker", checks["docker_cli"]["fix"])
        run.assert_not_called()

    def test_daemon_timeout_does_not_hide_compose_result(self) -> None:
        def command(args, **kwargs):
            self.assertEqual(kwargs["timeout"], 10)
            if "info" in args:
                raise subprocess.TimeoutExpired(args, 10)
            return SimpleNamespace(returncode=0, stdout="2.40.0\n", stderr="")

        with patch("inspect_rewind.doctor.shutil.which", return_value="/bin/docker"), \
             patch("inspect_rewind.doctor.subprocess.run", side_effect=command):
            report = doctor()
        checks = {check["name"]: check for check in report["checks"]}
        self.assertFalse(checks["docker_daemon"]["ok"])
        self.assertIn("10 seconds", checks["docker_daemon"]["detail"])
        self.assertTrue(checks["docker_compose"]["ok"])

    def test_missing_plugin_is_not_masked_by_importing_source(self) -> None:
        with patch("inspect_rewind.doctor.importlib.metadata.entry_points", return_value=[]), \
             patch("inspect_rewind.doctor.shutil.which", return_value=None):
            report = doctor()
        plugin = next(check for check in report["checks"] if check["name"] == "plugin")
        self.assertFalse(plugin["ok"])
        self.assertIn("same Python environment", plugin["fix"])


class InspectSmokeTests(unittest.TestCase):
    def test_failed_prerequisites_leave_report_and_preserve_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"INSPECT_REWIND_DIR": "original", "INSPECT_REWIND_CAPTURE": "svc:/data"}), \
             patch("tests.internal.smoke_inspect.doctor", return_value={"ok": False, "checks": []}), \
             patch("tests.internal.smoke_inspect.inspect_eval") as evaluate:
            report = run_check(Path(tmp))
            self.assertFalse(report["ok"])
            self.assertIn("Prerequisites", report["error"])
            self.assertEqual(json.loads(Path(report["report_file"]).read_text()), report)
            self.assertEqual(os.environ["INSPECT_REWIND_DIR"], "original")
            self.assertEqual(os.environ["INSPECT_REWIND_CAPTURE"], "svc:/data")
            evaluate.assert_not_called()

    def test_evaluation_failure_restores_configuration_and_keeps_report(self) -> None:
        def fail(*args, **kwargs):
            self.assertNotIn("INSPECT_REWIND_CAPTURE", os.environ)
            self.assertNotEqual(os.environ["INSPECT_REWIND_DIR"], "original")
            raise RuntimeError("Container startup failed")

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"INSPECT_REWIND_DIR": "original", "INSPECT_REWIND_CAPTURE": "svc:/data"}), \
             patch("tests.internal.smoke_inspect.doctor", return_value={"ok": True, "checks": []}), \
             patch("tests.internal.smoke_inspect.inspect_eval", side_effect=fail):
            report = run_check(Path(tmp))
            self.assertFalse(report["ok"])
            self.assertIn("Container startup failed", report["error"])
            self.assertEqual(os.environ["INSPECT_REWIND_DIR"], "original")
            self.assertEqual(os.environ["INSPECT_REWIND_CAPTURE"], "svc:/data")
            self.assertEqual(json.loads(Path(report["report_file"]).read_text()), report)

    @unittest.skipUnless(os.environ.get("RUN_REWIND_DOCKER") == "1", "set RUN_REWIND_DOCKER=1 for Docker integration")
    def test_smoke_check_runs_outside_checkout(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                report = run_check(Path(tmp) / "artifacts")
            finally:
                os.chdir(old_cwd)
            self.assertTrue(report["ok"], json.dumps(report, indent=2))
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["step"], 1)
            self.assertTrue(Path(report["parent_log"]).exists())
            self.assertTrue(Path(report["branch_log"]).exists())
            self.assertTrue(report["branch_checkpoints"])


if __name__ == "__main__":
    unittest.main()
