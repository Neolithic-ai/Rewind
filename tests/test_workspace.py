"""Framework-independent API contracts and an optional real Docker roundtrip."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rewind import Checkpoint, Environment
from rewind.env import restore, tar_digest


def archive(path: Path, value: bytes = b"seven") -> None:
    with tarfile.open(path, "w") as output:
        info = tarfile.TarInfo("data/value.txt")
        info.size = len(value)
        output.addfile(info, io.BytesIO(value))


class Workspace(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.compose = self.root / "compose.yaml"
        self.compose.write_text("services:\n  default:\n    image: debian:bookworm-slim\n")
        self.environment = Environment.from_compose(self.compose, capture={"default": ["/data"]})

    def checkpoint(self) -> Checkpoint:
        output = self.root / "checkpoint"
        output.mkdir()
        archive(output / "default__data.tar")
        (output / "meta.json").write_text(json.dumps({
            "kind": "rewind.checkpoint", "schema_version": 1, "backend": "filesystem",
            "compose_file": str(self.compose), "project": self.environment.project,
            "capture": {"default": ["/data"]}, "metadata": {"tool_call": "call-7"},
            "messages": [{"role": "user", "content": "continue"}],
            "content_sha256": {"default__data.tar": tar_digest(output / "default__data.tar")},
        }))
        return Checkpoint.load(output)

    def test_unique_projects_and_no_docker_during_configuration(self) -> None:
        with patch("rewind.workspace._run") as command:
            other = Environment.from_compose(self.compose, capture={"default": ["/data"]})
        command.assert_not_called()
        self.assertNotEqual(other.project, self.environment.project)
        self.assertFalse(other.started)

    def test_invalid_and_ambiguous_capture_paths_are_rejected(self) -> None:
        for capture in ({}, {"default": ["/"]}, {"default": ["relative"]},
                        {"default": ["/data/../root"]}, {"default": ["/a_b", "/a/b"]}):
            with self.subTest(capture=capture), self.assertRaises(ValueError):
                Environment.from_compose(self.compose, capture=capture)
        valid = Environment.from_compose(self.compose, capture={"default": ["/data", "/data/sub"]})
        self.assertEqual(valid.capture, {"default": ["/data"]})

    def test_live_handle_save_load_preserves_identity_and_requires_valid_state(self) -> None:
        self.environment._started = True
        self.environment._owned = True
        target = self.environment.save(self.root / "handle.json")
        loaded = Environment.load(target)
        self.assertEqual(loaded.project, self.environment.project)
        self.assertEqual(loaded.capture, self.environment.capture)
        self.assertTrue(loaded.started)
        state = json.loads(target.read_text())
        del state["project"]
        target.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "project"):
            Environment.load(target)

    def test_argv_passes_directly_without_expanding_shell_syntax(self) -> None:
        self.environment._started = True
        argv = ["printf", "%s", "$(do-not-run) a b"]
        with patch.object(self.environment._engine, "exec", return_value=(0, "ok")) as execute:
            self.assertEqual(self.environment.exec("default", argv), (0, "ok"))
        self.assertEqual(execute.call_args.args[1], argv)

    def test_shell_strings_use_sh_not_bash(self) -> None:
        self.environment._started = True
        with patch.object(self.environment._engine, "exec", return_value=(0, "ok")) as execute:
            self.environment.exec("default", "printf ok")
        self.assertEqual(execute.call_args.args[1], ["sh", "-c", "printf ok"])

    def test_cleanup_errors_are_reported_without_losing_original_failure(self) -> None:
        self.environment._started = True
        self.environment._owned = True
        with patch.object(self.environment._engine, "down", side_effect=RuntimeError("Docker unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Docker unavailable"):
                self.environment.down()
            self.assertTrue(self.environment.started)
            original = RuntimeError("original failure")
            self.environment._cleanup_failed(original)
            self.assertIn("cleanup", original.__notes__[0])
            self.assertIn("Docker unavailable", original.__notes__[0])

    def test_corrupted_archive_is_rejected_before_docker(self) -> None:
        checkpoint = self.checkpoint()
        archive(checkpoint.path / "default__data.tar", b"changed")
        with patch("rewind.workspace._run") as command:
            with self.assertRaisesRegex(ValueError, "hash verification"):
                checkpoint.branch()
        command.assert_not_called()

    def test_missing_archive_is_rejected_before_docker(self) -> None:
        checkpoint = self.checkpoint()
        (checkpoint.path / "default__data.tar").unlink()
        with patch("rewind.workspace._run") as command:
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                checkpoint.branch()
        command.assert_not_called()

    def test_relocated_checkpoint_accepts_compose_override_and_keeps_messages(self) -> None:
        checkpoint = self.checkpoint()
        original_messages = checkpoint.messages
        moved = self.root / "moved"
        shutil.move(checkpoint.path, moved)
        self.compose.unlink()
        replacement = self.root / "replacement.yaml"
        replacement.write_text("services: {default: {image: debian:bookworm-slim}}")
        checkpoint = Checkpoint.load(moved)
        with patch.object(Environment, "_prepare"), patch("rewind.workspace.restore", return_value=(0.1, {"ok": True})):
            branch = checkpoint.branch(compose_file=replacement)
        self.assertEqual(branch.compose_file, replacement)
        self.assertNotEqual(branch.project, self.environment.project)
        self.assertEqual(checkpoint.messages, original_messages)
        self.assertEqual(checkpoint.metadata, {"tool_call": "call-7"})
        self.assertFalse(branch.restore_report["capabilities"]["sessions"])
        self.assertIn("not restored", checkpoint.restore_notice)

    def test_failed_branch_cleans_its_project(self) -> None:
        checkpoint = self.checkpoint()
        with patch.object(Environment, "_prepare"), patch("rewind.workspace.restore", side_effect=RuntimeError("restore failed")), patch("rewind.workspace.Env.down") as cleanup:
            with self.assertRaisesRegex(RuntimeError, "restore failed"):
                checkpoint.branch()
        cleanup.assert_called_once()

    def test_existing_project_is_never_adopted_or_cleaned(self) -> None:
        config = {"services": {"default": {}}}
        results = [SimpleNamespace(stdout=json.dumps(config).encode()), SimpleNamespace(stdout=b"existing-container")]
        with patch("rewind.workspace._run", side_effect=results), patch.object(self.environment._engine, "down") as cleanup:
            with self.assertRaisesRegex(ValueError, "already has resources"):
                self.environment.start()
            # User cleanup in a finally block must never remove the pre-existing project.
            self.environment.down()
        cleanup.assert_not_called()

    def test_unstarted_saved_handle_does_not_own_resources(self) -> None:
        target = self.environment.save(self.root / "unused-handle.json")
        loaded = Environment.load(target)
        with patch.object(loaded._engine, "down") as cleanup:
            loaded.down()
        cleanup.assert_not_called()

    def test_loaded_live_handle_can_remove_its_own_resources(self) -> None:
        self.environment._started = self.environment._owned = True
        loaded = Environment.load(self.environment.save(self.root / "live.json"))
        with patch.object(loaded._engine, "down") as cleanup:
            loaded.down()
        cleanup.assert_called_once_with(check=True)
        self.assertFalse(loaded.started)

    def test_shared_namespaces_and_storage_are_rejected(self) -> None:
        configurations = [
            {"services": {"default": {setting: mode}}}
            for setting in ("network_mode", "pid", "ipc")
            for mode in ("host", "container:external")
        ]
        configurations.extend([
            {"services": {"default": {"volumes_from": ["container:external"]}}},
            {"services": {"default": {}}, "volumes": {"data": {"driver_opts": {"device": "/shared", "o": "bind", "type": "none"}}}},
            {"services": {"default": {}}, "volumes": {"data": {"driver": "remote-storage"}}},
        ])
        for config in configurations:
            with self.subTest(config=config), patch("rewind.workspace._run", return_value=SimpleNamespace(stdout=json.dumps(config).encode())):
                with self.assertRaises(ValueError):
                    self.environment.start()
                self.assertFalse(self.environment._owned)

    def test_writable_shared_bind_mount_is_refused(self) -> None:
        config = {"services": {"default": {"volumes": [{"type": "bind", "target": "/data"}]}}}
        with patch("rewind.workspace._run", return_value=SimpleNamespace(stdout=json.dumps(config).encode())):
            with self.assertRaisesRegex(ValueError, "writable bind mount"):
                self.environment.start()

    def test_failed_snapshot_does_not_publish_partial_checkpoint(self) -> None:
        self.environment._started = True
        output = self.root / "failed"
        def fail_snapshot(engine, directory, metadata):
            directory.mkdir()
            (directory / "partial.tar").write_bytes(b"partial")
            raise RuntimeError("copy failed")
        with patch.object(self.environment._engine, "container_id", return_value="container"), patch("rewind.workspace._run", return_value=SimpleNamespace(returncode=0)), patch("rewind.workspace.snapshot", side_effect=fail_snapshot):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                self.environment.checkpoint(output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".failed-*")), [])

    def test_strict_restore_never_starts_services_after_hash_mismatch(self) -> None:
        checkpoint = self.checkpoint()
        changed = self.root / "changed.tar"
        archive(changed, b"incorrect restored bytes")
        engine = Mock()
        engine.scenario.services = {"default": SimpleNamespace(name="default", capture=["/data"])}
        engine.cp_out.side_effect = lambda service, path, destination: shutil.copyfile(changed, destination)
        with self.assertRaisesRegex(RuntimeError, "initially empty dedicated named volumes"):
            restore(engine, checkpoint.path, verify=True)
        engine.start.assert_not_called()

    def test_strict_restore_missing_parent_error_explains_supported_layout(self) -> None:
        checkpoint = self.checkpoint()
        engine = Mock()
        engine.scenario.services = {"default": SimpleNamespace(name="default", capture=["/data"])}
        engine.cp_in.side_effect = RuntimeError("destination does not exist")
        with self.assertRaisesRegex(RuntimeError, "must already exist in the fresh image"):
            restore(engine, checkpoint.path, verify=True)
        engine.start.assert_not_called()


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(docker_available(), "requires Docker")
class DockerWorkspace(unittest.TestCase):
    def test_real_parent_and_two_branches_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose = root / "compose.yaml"
            compose.write_text("""services:
  default:
    image: debian:bookworm-slim
    command: [sleep, infinity]
    init: true
    volumes:
      - state:/data
volumes:
  state:
""")
            environments = []
            try:
                parent = Environment.from_compose(compose, capture={"default": ["/data"]}).start()
                environments.append(parent)
                self.assertEqual(parent.exec("default", "printf 7 > /data/count")[0], 0)
                messages = [{"role": "assistant", "tool": "write", "result": "7"}]
                checkpoint = parent.checkpoint(root / "saved", messages=messages, metadata={"step": 1})
                self.assertEqual(parent.exec("default", "printf 107 > /data/count")[0], 0)
                for value in (8, 17):
                    branch = Checkpoint.load(checkpoint.path).branch()
                    environments.append(branch)
                    self.assertTrue(branch.restore_report["ok"])
                    self.assertEqual(branch.exec("default", ["cat", "/data/count"]), (0, "7"))
                    self.assertEqual(branch.exec("default", f"printf {value} > /data/count")[0], 0)
                self.assertEqual([env.exec("default", "cat /data/count")[1] for env in environments],
                                 ["107", "8", "17"])
                self.assertEqual(Checkpoint.load(checkpoint.path).messages, messages)
                self.assertEqual(Checkpoint.load(checkpoint.path).metadata, {"step": 1})
            finally:
                for environment in reversed(environments):
                    environment.down()


if __name__ == "__main__":
    unittest.main()
