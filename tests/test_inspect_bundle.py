"""Portable Inspect checkpoints remain forkable after all source paths disappear."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.event import InfoEvent
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, read_eval_log, write_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser, get_model
from inspect_ai.tool import ToolCall
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_rewind.bundle import export_bundle, import_bundle
from inspect_rewind.cli import main
from inspect_rewind.fork import fork, mock_outputs, snapshots
from rewind.bundle import extracted_bundle, write_bundle
from rewind.env import tar_digest


def fake_compose(source: Path, destination: Path, **kwargs) -> Path:
    destination.mkdir(parents=True)
    target = destination / "compose.yaml"
    shutil.copy2(source, target)
    return target


class InspectBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "task.py").write_text("raise AssertionError('task executed during export or import')\n")
        (self.source / "compose.yaml").write_text("services:\n  default:\n    image: busybox\n")
        (self.source / ".env").write_text("DO_NOT_PACKAGE_THIS=secret\n")
        (self.source / "unrelated.txt").write_text("not implicitly selected\n")
        messages = [ChatMessageUser(content="Do three things.")]
        events = [InfoEvent(source="rewind", data={"rewind": "capture", "capture": {"default": ["/workspace"]}})]
        self.entries = {}
        for number in range(1, 4):
            directory = self.source / "snapshots" / f"step-{number:03d}"
            directory.mkdir(parents=True)
            archive = directory / "default__workspace.tar"
            with tarfile.open(archive, "w") as output:
                data = f"step {number}".encode()
                info = tarfile.TarInfo("workspace/state.txt")
                info.size = len(data)
                output.addfile(info, io.BytesIO(data))
            hashes = {archive.name: tar_digest(archive)}
            (directory / "meta.json").write_text(json.dumps({"content_sha256": hashes}))
            entry = {"step": number, "tool_call_id": f"call-{number}",
                     "snapshot_dir": str(directory), "content_sha256": hashes}
            self.entries[number] = entry
            events.append(InfoEvent(source="rewind", data={"rewind": "snapshot", **entry}))
            messages.extend([
                ChatMessageAssistant(content="", tool_calls=[ToolCall(
                    id=f"call-{number}", function="bash", arguments={"command": "true"})]),
                ChatMessageTool(content="ok", tool_call_id=f"call-{number}")])
        log = EvalLog(eval=EvalSpec(
            created="2026-01-01T00:00:00Z", task="task", task_file=str(self.source / "task.py"),
            model="mockllm/model", dataset=EvalDataset(), config=EvalConfig(),
            sandbox=SandboxEnvironmentSpec(type="rewind", config=str(self.source / "compose.yaml"))),
            samples=[EvalSample(id="sample", epoch=1, input="Do three things.", target="done",
                                messages=messages, events=events)])
        self.log_file = self.source / "parent.eval"
        write_eval_log(log, self.log_file)
        self.compose_patch = patch("inspect_rewind.bundle.stage_compose", side_effect=fake_compose)
        self.compose_patch.start()

    def tearDown(self) -> None:
        self.compose_patch.stop()
        self.temporary.cleanup()

    def test_cli_roundtrip_relocates_selected_and_ancestor_checkpoints_without_running_task(self) -> None:
        archive = self.root / "run.rewind"
        with patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(main(["export", str(self.log_file), "--step", "2", "--output", str(archive)]), 0)
        with extracted_bundle(archive) as (staging, manifest):
            self.assertEqual(manifest["checkpoints"], [1, 2])
            self.assertFalse((staging / "task" / ".env").exists())
            self.assertFalse((staging / "task" / "unrelated.txt").exists())
        shutil.rmtree(self.source)
        destination = self.root / "new-location"
        with patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(main(["import", str(archive), "--output", str(destination)]), 0)
        log = read_eval_log(str(destination / "parent.eval"), resolve_attachments=True)
        self.assertEqual(log.eval.task_file, str(destination / "task" / "task.py"))
        self.assertEqual(log.eval.sandbox.config, str(destination / "environment" / "compose.yaml"))
        entries = snapshots(log.samples[0])
        self.assertEqual(sorted(entries), [1, 2])
        for step, entry in entries.items():
            self.assertEqual(entry["content_sha256"], self.entries[step]["content_sha256"])
            self.assertTrue((Path(entry["snapshot_dir"]) / "meta.json").is_file())

    def test_explicit_sources_include_local_dependencies_and_exclude_private_files(self) -> None:
        (self.source / "task.py").write_text("import helper\n")
        (self.source / "helper.py").write_text("VALUE = 42\n")
        (self.source / "key.pem").write_text("not a source file")
        archive = self.root / "run.rewind"
        with self.assertRaisesRegex(ValueError, "--task-dir"):
            export_bundle(self.log_file, archive, step=2)
        export_bundle(self.log_file, archive, step=2, task_dir=self.source)
        with extracted_bundle(archive) as (staging, manifest):
            self.assertTrue((staging / "task" / "helper.py").is_file())
            self.assertFalse((staging / "task" / ".env").exists())
            self.assertFalse((staging / "task" / "key.pem").exists())
            self.assertFalse((staging / "task" / "parent.eval").exists())

    def test_corrupted_checkpoint_rejected_before_archive_published(self) -> None:
        (Path(self.entries[2]["snapshot_dir"]) / "meta.json").write_text('{"content_sha256": {}}')
        archive = self.root / "run.rewind"
        with self.assertRaisesRegex(ValueError, "transcript hashes"):
            export_bundle(self.log_file, archive, step=2)
        self.assertFalse(archive.exists())

    def test_missing_ancestor_is_reported_and_removed_from_forkable_index(self) -> None:
        shutil.rmtree(self.entries[1]["snapshot_dir"])
        archive = export_bundle(self.log_file, self.root / "run.rewind", step=2)
        with extracted_bundle(archive) as (staging, manifest):
            self.assertEqual(manifest["unavailable_ancestor_steps"], [1])
        restored = import_bundle(archive, self.root / "restored")
        log = read_eval_log(str(restored), resolve_attachments=True)
        self.assertEqual(sorted(snapshots(log.samples[0])), [2])

    def test_sample_config_and_metadata_override_task_defaults(self) -> None:
        config = self.source / "sample.yaml"
        config.write_text("services:\n  default:\n    image: busybox\n    environment:\n      VARIANT: ${SAMPLE_METADATA_VARIANT}\n")
        log = read_eval_log(str(self.log_file), resolve_attachments=True)
        log.samples[0].sandbox = SandboxEnvironmentSpec(type="docker", config="sample.yaml")
        log.samples[0].metadata = {"variant": "sample-value"}
        write_eval_log(log, self.log_file)
        with patch("inspect_rewind.bundle.stage_compose", side_effect=fake_compose) as stage:
            archive = export_bundle(self.log_file, self.root / "run.rewind", step=2)
        self.assertEqual(stage.call_args.args[0], config)
        self.assertEqual(stage.call_args.kwargs["environment"], {"SAMPLE_METADATA_VARIANT": "sample-value"})
        imported = import_bundle(archive, self.root / "restored")
        restored = read_eval_log(str(imported), resolve_attachments=True)
        self.assertEqual(restored.samples[0].sandbox.config, restored.eval.sandbox.config)

    def test_import_refuses_outside_file_references_even_with_valid_inventory(self) -> None:
        archive = export_bundle(self.log_file, self.root / "run.rewind", step=2)
        with extracted_bundle(archive) as (staging, manifest):
            (staging / "bundle.json").unlink()
            manifest["task_file"] = "../outside.py"
            write_bundle(staging, self.root / "invalid.rewind", manifest)
        destination = self.root / "restored"
        with self.assertRaisesRegex(ValueError, "invalid bundle file"):
            import_bundle(self.root / "invalid.rewind", destination)
        self.assertFalse(destination.exists())


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(docker_available(), "needs Docker")
class InspectBundleDockerTests(unittest.TestCase):
    def test_relocated_bundle_forks_and_branch_bundle_retains_inherited_steps(self) -> None:
        fixtures = Path(__file__).parent / "internal" / "plain_task"
        with tempfile.TemporaryDirectory(prefix="inspect-bundle-live-") as temporary:
            root = Path(temporary)
            source = root / "original"
            source.mkdir()
            for name in ("task.py", "compose.yaml"):
                shutil.copy2(fixtures / name, source / name)
            (source / "input.txt").write_text("portable-asset\n")
            config = yaml.safe_load((source / "compose.yaml").read_text())
            config["services"]["default"]["volumes"].append("./input.txt:/opt/input:ro")
            config["services"]["default"]["configs"] = [{"source": "input", "target": "/opt/config"}]
            config["services"]["default"]["environment"] = {"VARIANT": "${SAMPLE_METADATA_VARIANT:-missing}"}
            config["configs"] = {"input": {"file": "./input.txt"}}
            (source / "sample.yaml").write_text(yaml.safe_dump(config))
            task_source = (source / "task.py").read_text()
            task_source = task_source.replace(
                'target="3")', 'target="3", sandbox=("docker", "sample.yaml"), metadata={"variant": "sample-value"})')
            (source / "task.py").write_text(task_source)
            with patch.dict(os.environ, {"INSPECT_REWIND_DIR": str(root / "original-snapshots")}):
                model = get_model("mockllm/parent", custom_outputs=mock_outputs(
                    yaml.safe_load((fixtures / "parent_mock.yaml").read_text())))
                parent = inspect_eval(f"{source / 'task.py'}@plain_task", sandbox="rewind", model=model,
                                      log_dir=str(root / "original-logs"), display="none")[0]
                self.assertEqual(parent.status, "success")
                archive = export_bundle(parent.location, root / "parent.rewind", step=2)
            shutil.rmtree(source)
            shutil.rmtree(root / "original-snapshots")
            shutil.rmtree(root / "original-logs")
            imported = import_bundle(archive, root / "imported")
            with patch.dict(os.environ, {"INSPECT_REWIND_DIR": str(root / "branch-snapshots")}):
                commands = yaml.safe_load((fixtures / "branch_mock.yaml").read_text())
                commands[0]["bash"] += ('; test "$(cat /opt/input)" = portable-asset '
                                         '&& test "$(cat /opt/config)" = portable-asset '
                                         '&& test "$VARIANT" = sample-value && echo asset-check-passed')
                branch, report = fork(imported, None, 2, model="mockllm/other", outputs=mock_outputs(commands),
                    log_dir=str(root / "branch-logs"))
                self.assertEqual(branch.status, "success")
                self.assertTrue(report["restore_ok"] and report["prefix_ok"])
                self.assertTrue(any("asset-check-passed" in message.text for message in branch.samples[0].messages
                                    if isinstance(message, ChatMessageTool)))
                second_archive = export_bundle(branch.location, root / "branch.rewind", step=3)
            shutil.rmtree(root / "imported")
            shutil.rmtree(root / "branch-snapshots")
            shutil.rmtree(root / "branch-logs")
            second_import = import_bundle(second_archive, root / "imported-again")
            with patch.dict(os.environ, {"INSPECT_REWIND_DIR": str(root / "continuation-snapshots")}):
                for step in (1, 3):
                    with self.subTest(step=step):
                        continuation, report = fork(second_import, None, step, model="mockllm/third",
                                                    outputs=mock_outputs([{"bash": "cat /srv/notes/todo.txt /opt/input /opt/config"},
                                                                          {"submit": "3"}]))
                        self.assertEqual(continuation.status, "success")
                        self.assertTrue(report["restore_ok"] and report["prefix_ok"])


if __name__ == "__main__":
    unittest.main()
