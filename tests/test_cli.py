"""Checkpoint discovery and relocation should not need model calls or Docker."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inspect_ai.event import InfoEvent
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolCall
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_rewind.cli import checkpoint_report, main
from inspect_rewind.fork import _load_task, find_sample, fork


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "step-001"
        self.checkpoint.mkdir()
        (self.checkpoint / "meta.json").write_text("{}")
        self.entry = {"step": 1, "tool_call_id": "call1", "snapshot_dir": str(self.checkpoint)}
        self.sample = SimpleNamespace(id="sample", epoch=1, sandbox=None, messages=[
            ChatMessageAssistant(content="", tool_calls=[ToolCall(
                id="call1", function="bash", arguments={"command": "echo one"})]),
            ChatMessageTool(content="one", tool_call_id="call1")], events=[
                InfoEvent(source="rewind", data={"rewind": "snapshot", **self.entry})])
        self.log = SimpleNamespace(location=str(self.root / "logs" / "parent.eval"),
                                   samples=[self.sample], eval=SimpleNamespace(
                                       model="mockllm/model", task="task", task_registry_name="task",
                                       task_file="tasks/task.py", task_args={}))

    def test_single_sample_is_selected_but_ambiguous_selection_is_actionable(self):
        self.assertIs(find_sample(self.log), self.sample)
        self.log.samples.append(SimpleNamespace(id="other", epoch=1))
        with self.assertRaisesRegex(ValueError, "choose --sample.*sample, other"):
            find_sample(self.log)

    def test_listing_reports_missing_files_and_tool_arguments(self):
        with patch("inspect_ai.log.read_eval_log", return_value=self.log):
            report = checkpoint_report("parent.eval")
            row = report["checkpoints"][0]
            self.assertEqual(row["arguments"], {"command": "echo one"})
            self.assertTrue(row["available"] and row["forkable"])
            (self.checkpoint / "meta.json").unlink()
            self.assertFalse(checkpoint_report("parent.eval")["checkpoints"][0]["available"])

    def test_json_listing_is_parseable(self):
        out = io.StringIO()
        with patch("inspect_ai.log.read_eval_log", return_value=self.log), contextlib.redirect_stdout(out):
            result = main(["checkpoints", "parent.eval", "--json"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(out.getvalue())["checkpoints"][0]["sample"], "sample")

    def test_missing_checkpoint_fails_before_loading_task_or_model(self):
        (self.checkpoint / "meta.json").unlink()
        with patch("inspect_rewind.fork.read_eval_log", return_value=self.log), \
                patch("inspect_rewind.fork._load_task") as load, \
                patch("inspect_rewind.fork.resolve_model") as model:
            with self.assertRaisesRegex(FileNotFoundError, "--snapshot-dir"):
                fork("parent.eval", None, 1)
        load.assert_not_called()
        model.assert_not_called()

    def test_missing_log_has_concise_cli_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            result = main(["checkpoints", str(self.root / "missing.eval")])
        self.assertEqual(result, 2)
        self.assertNotIn("Traceback", err.getvalue())

    def test_wrong_intact_checkpoint_is_refused_before_task_or_model(self):
        self.entry["content_sha256"] = {"path.tar": "expected"}
        self.sample.events[0].data["content_sha256"] = self.entry["content_sha256"]
        moved = self.root / "different-step"
        moved.mkdir()
        (moved / "meta.json").write_text(json.dumps({"content_sha256": {"path.tar": "different"}}))
        with patch("inspect_rewind.fork.read_eval_log", return_value=self.log), \
                patch("inspect_rewind.fork._load_task") as load:
            with self.assertRaisesRegex(ValueError, "does not match the saved hashes"):
                fork("parent.eval", None, 1, snapshot_dir=moved)
        load.assert_not_called()

    def test_relocated_task_uses_its_own_sandbox_configuration(self):
        self.sample.uuid = "sample-uuid"
        self.log.plan = None
        self.log.eval.eval_id = "parent-id"
        self.log.eval.tags = []
        self.log.eval.metadata = {}
        self.log.eval.sandbox = SimpleNamespace(config="/old-machine/compose.yaml")
        task = Task(dataset=[Sample(id="sample", input="test")],
                    sandbox=("docker", "/new-machine/compose.yaml"))
        result = SimpleNamespace(location="branch.eval", status="success", samples=[])
        with patch("inspect_rewind.fork.read_eval_log", side_effect=[self.log, result]), \
                patch("inspect_rewind.fork._load_task", return_value=task), \
                patch("inspect_rewind.fork.inspect_eval", return_value=[result]) as evaluate:
            fork("parent.eval", None, 1, task_spec="/new-machine/task.py@task")
        self.assertEqual(evaluate.call_args.kwargs["sandbox"], "rewind")
        self.assertEqual(evaluate.call_args.args[0].sandbox.config, "/new-machine/compose.yaml")

    def test_logged_relative_sandbox_is_resolved_before_becoming_a_sample_override(self):
        self.sample.uuid = "sample-uuid"
        self.log.plan = None
        self.log.eval.eval_id = "parent-id"
        self.log.eval.tags = []
        self.log.eval.metadata = {}
        source = self.root / "tasks"
        source.mkdir()
        (source / "task.py").write_text("# task source")
        self.log.eval.sandbox = SandboxEnvironmentSpec(type="rewind", config="compose.yaml")
        task = Task(dataset=[Sample(id="sample", input="test")], sandbox=("docker", "compose.yaml"))
        result = SimpleNamespace(location="branch.eval", status="success", samples=[])
        for config in (None, "sample.yaml"):
            with self.subTest(sample_config=config):
                self.sample.sandbox = SandboxEnvironmentSpec(type="docker", config=config) if config else None
                with patch("inspect_rewind.fork.read_eval_log", side_effect=[self.log, result]), \
                        patch("inspect_rewind.fork._load_task", return_value=task), \
                        patch("inspect_rewind.fork.inspect_eval", return_value=[result]) as evaluate:
                    fork("parent.eval", None, 1)
                expected = str(source / (config or "compose.yaml"))
                self.assertEqual(evaluate.call_args.kwargs["sandbox"], ("rewind", expected))
                self.assertEqual(evaluate.call_args.args[0].dataset[0].sandbox.config, expected)

    def test_relative_task_resolves_from_log_ancestors_outside_checkout(self):
        task = self.root / "tasks" / "task.py"
        task.parent.mkdir()
        task.write_text("# test loader path")
        other = self.root / "elsewhere"
        other.mkdir()
        before = Path.cwd()
        try:
            os.chdir(other)
            with patch("inspect_rewind.fork.load_tasks", return_value=["loaded"]) as load:
                self.assertEqual(_load_task(self.log), "loaded")
                self.assertEqual(load.call_args.args[0], [f"{task}@task"])
                _load_task(self.log, "custom.py@custom")
                self.assertEqual(load.call_args.args[0], ["custom.py@custom"])
        finally:
            os.chdir(before)

    def test_bad_mock_yaml_is_rejected_before_fork(self):
        spec = self.root / "bad.yaml"
        for content in ("bash: echo wrong-shape", "- {}", "- unknown: value"):
            spec.write_text(content)
            with patch("inspect_rewind.fork.fork") as branch, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["fork", "unused.eval", "--step", "1", "--mock-outputs", str(spec)]), 2)
            branch.assert_not_called()
