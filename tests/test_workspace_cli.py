"""Command parsing and lifecycle safeguards for the framework-independent CLI."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rewind.cli import main
from tests.internal.smoke_environment import run_check


class WorkspaceCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = types.ModuleType("rewind.workspace")
        self.module.Environment = MagicMock()
        self.module.Checkpoint = MagicMock()
        self.env = MagicMock(project="rewind-test", compose_file=Path("compose.yaml"),
                             capture={"default": ["/root"]}, restore_report=None)
        self.module.Environment.load.return_value = self.env
        self.module.Environment.from_compose.return_value = self.env
        self.patch = patch.dict(sys.modules, {"rewind.workspace": self.module})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def invoke(self, args: list[str]) -> tuple[int, dict]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(args)
        return code, json.loads(stdout.getvalue() or stderr.getvalue())

    def test_exec_preserves_command_arguments_and_failure_status(self) -> None:
        self.env.exec.return_value = (7, "did not succeed\n")
        code, report = self.invoke([
            "exec", "state.json", "--service", "worker", "--",
            "bash", "-c", "printf '%s' '$HOME'; exit 7", "--service", "command-flag",
        ])
        self.assertEqual(code, 7)
        self.assertFalse(report["ok"])
        self.env.exec.assert_called_once_with(
            "worker", ["bash", "-c", "printf '%s' '$HOME'; exit 7", "--service", "command-flag"])

    def test_up_refuses_to_replace_existing_environment_handle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            state.write_text("original")
            code, report = self.invoke([
                "up", "compose.yaml", "--capture", "default:/root", "--state", str(state),
            ])
            self.assertEqual(code, 1)
            self.assertIn("already exists", report["error"])
            self.assertEqual(state.read_text(), "original")
            self.module.Environment.from_compose.assert_not_called()

    def test_state_write_failure_cleans_up_new_environment(self) -> None:
        self.env.save.side_effect = OSError("disk full")
        with tempfile.TemporaryDirectory() as tmp:
            code, report = self.invoke([
                "up", "compose.yaml", "--capture", "default:/root", "--state", str(Path(tmp) / "state.json"),
            ])
        self.assertEqual(code, 1)
        self.assertIn("disk full", report["error"])
        self.env.start.assert_called_once()
        self.env.down.assert_called_once()

    def test_invalid_capture_fails_before_starting_environment(self) -> None:
        code, report = self.invoke(["up", "compose.yaml", "--capture", "/root", "--state", "unused.json"])
        self.assertEqual(code, 1)
        self.assertIn("SERVICE:/absolute/container/path", report["error"])
        self.module.Environment.from_compose.assert_not_called()

    def test_down_persists_released_ownership_for_later_cli_calls(self) -> None:
        code, report = self.invoke(["down", "state.json"])
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "removed")
        self.env.down.assert_called_once()
        self.env.save.assert_called_once_with(Path("state.json"))

    def test_import_does_not_load_inspect(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import rewind.cli, sys; assert 'inspect_ai' not in sys.modules"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_exposes_only_environment_commands(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as result:
            main(["--help"])
        self.assertEqual(result.exception.code, 0)
        for command in ("doctor", "up", "checkpoint", "branch", "export", "import", "exec", "down"):
            self.assertIn(command, output.getvalue())
        self.assertNotIn("demo", output.getvalue())


class WorkspaceSmokeTests(unittest.TestCase):
    def test_failed_checkpoint_cleans_parent_and_persists_failure(self) -> None:
        environment = MagicMock(project="rewind-smoke-failure")
        environment.exec.return_value = (0, "7")
        environment.checkpoint.side_effect = RuntimeError("snapshot failed")
        with tempfile.TemporaryDirectory() as tmp, \
             patch("rewind.cli._doctor", return_value={"ok": True, "checks": []}), \
             patch("tests.internal.smoke_environment.Environment.from_compose", return_value=environment), \
             patch("tests.internal.smoke_environment._removed", return_value=True):
            report = run_check(Path(tmp))
            self.assertFalse(report["ok"])
            self.assertIn("snapshot failed", report["error"])
            self.assertTrue(report["checks"]["environments_removed"])
            self.assertEqual(json.loads(Path(report["report_file"]).read_text()), report)
        environment.down.assert_called_once()

    def test_cleanup_failure_is_recorded_after_run_failure(self) -> None:
        environment = MagicMock(project="rewind-smoke-failure")
        environment.exec.return_value = (0, "7")
        environment.checkpoint.side_effect = RuntimeError("snapshot failed")
        environment.down.side_effect = RuntimeError("daemon unavailable")
        with tempfile.TemporaryDirectory() as tmp, \
             patch("rewind.cli._doctor", return_value={"ok": True, "checks": []}), \
             patch("tests.internal.smoke_environment.Environment.from_compose", return_value=environment):
            report = run_check(Path(tmp))
        self.assertFalse(report["ok"])
        self.assertFalse(report["checks"]["environments_removed"])
        self.assertIn("daemon unavailable", report["cleanup"][0]["error"])


if __name__ == "__main__":
    unittest.main()
