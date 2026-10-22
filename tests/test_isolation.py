"""Branch storage must be private before Inspect creates any containers."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anyio
import yaml

from inspect_rewind.sandbox import RewindSandboxEnvironment, _validate_isolation
from rewind import Environment
from rewind.isolation import validate_compose_isolation


class IsolationPolicy(unittest.TestCase):
    def test_rejects_resources_shared_between_projects(self) -> None:
        configs = [
            {"services": {"default": {setting: mode}}}
            for setting in ("network_mode", "pid", "ipc")
            for mode in ("host", "container:parent")
        ]
        configs.extend([
            {"services": {"default": {"network_mode": "bridge"}}},
            {"services": {"default": {"network_mode": "shared-network"}}},
            {"services": {"default": {"container_name": "fixed-name"}}},
            {"services": {"default": {"volumes_from": ["container:parent:ro"]}}},
            {"services": {"default": {"volumes": [{"type": "bind", "target": "/data"}]}}},
            {"volumes": {"data": {"external": True, "name": "branch_data"}}},
            {"volumes": {"data": {"name": "parent_data"}}},
            {"volumes": {"data": {"driver_opts": {"device": "/shared", "type": "none", "o": "bind"}}}},
            {"volumes": {"data": {"driver": "remote-storage"}}},
            {"networks": {"default": {"external": True, "name": "branch_default"}}},
            {"networks": {"default": {"name": "parent_default"}}},
        ])
        for config in configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_compose_isolation(config, "branch")

    def test_allows_private_resources_and_read_only_inputs(self) -> None:
        validate_compose_isolation({
            "services": {
                "default": {
                    "volumes": [
                        {"type": "bind", "source": "/inputs", "target": "/input", "read_only": True},
                        {"type": "volume", "source": "data", "target": "/data"},
                    ],
                    "network_mode": "service:worker", "pid": "service:worker",
                    "ipc": "service:worker", "volumes_from": ["worker:ro"],
                },
                "worker": {"network_mode": "none"},
            },
            "volumes": {"data": {"name": "branch_data", "driver": "local"}},
            "networks": {"default": {"name": "branch_default"}},
        }, "branch")


class InspectIsolation(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "compose.yaml"
        self.config.write_text("services: {}")
        self.project = SimpleNamespace(name="inspect-isolation-ibbbbbb", config=str(self.config))

    def validate(self, config: dict) -> None:
        command = SimpleNamespace(success=True, stdout=json.dumps(config), stderr="")
        with patch("inspect_rewind.sandbox.ComposeProject.create", new_callable=AsyncMock, return_value=self.project), \
                patch("inspect_rewind.sandbox.compose_command", new_callable=AsyncMock, return_value=command):
            anyio.run(_validate_isolation, "task", str(self.config), {})

    def test_invalid_config_never_reaches_docker_startup_or_cleanup(self) -> None:
        config = {"volumes": {"data": {"name": "parent_data"}}}
        command = SimpleNamespace(success=True, stdout=json.dumps(config), stderr="")
        with patch("inspect_rewind.sandbox.ComposeProject.create", new_callable=AsyncMock, return_value=self.project), \
                patch("inspect_rewind.sandbox.compose_command", new_callable=AsyncMock, return_value=command) as compose, \
                patch("inspect_rewind.sandbox.DockerSandboxEnvironment.sample_init", new_callable=AsyncMock) as startup, \
                patch("inspect_ai.util._sandbox.docker.docker.project_cleanup", new_callable=AsyncMock) as cleanup:
            with self.assertRaisesRegex(ValueError, "shared across projects"):
                anyio.run(RewindSandboxEnvironment.sample_init, "task", str(self.config), {})
        startup.assert_not_called()
        cleanup.assert_not_called()
        self.assertEqual(compose.call_args.args[0], ["config", "--format", "json"])
        self.assertTrue(self.config.exists())

    def test_private_config_reaches_normal_inspect_startup(self) -> None:
        command = SimpleNamespace(success=True, stdout=json.dumps({
            "services": {"default": {}},
            "volumes": {"data": {"name": f"{self.project.name}_data"}},
        }), stderr="")
        with patch("inspect_rewind.sandbox.ComposeProject.create", new_callable=AsyncMock, return_value=self.project), \
                patch("inspect_rewind.sandbox.compose_command", new_callable=AsyncMock, return_value=command), \
                patch("inspect_rewind.sandbox.DockerSandboxEnvironment.sample_init", new_callable=AsyncMock,
                      side_effect=RuntimeError("normal startup")) as startup:
            with self.assertRaisesRegex(RuntimeError, "normal startup"):
                anyio.run(RewindSandboxEnvironment.sample_init, "task", str(self.config), {})
        startup.assert_awaited_once_with("task", str(self.config), {})

    def test_sample_metadata_is_used_during_preflight(self) -> None:
        self.config.write_text("services:\n  default:\n    image: '${SAMPLE_METADATA_IMAGE}'\n")
        command = SimpleNamespace(success=True, stdout='{"services": {"default": {}}}', stderr="")
        with patch("inspect_rewind.sandbox.ComposeProject.create", new_callable=AsyncMock, return_value=self.project) as create, \
                patch("inspect_rewind.sandbox.compose_command", new_callable=AsyncMock, return_value=command):
            anyio.run(_validate_isolation, "task", str(self.config), {"image": "debian:bookworm-slim"})
        self.assertEqual(create.call_args.kwargs["env"], {"SAMPLE_METADATA_IMAGE": "debian:bookworm-slim"})

    def test_generated_config_removed_even_when_validation_fails(self) -> None:
        generated = self.root / f"{self.project.name}.yaml"
        generated.write_text("services: {}")
        self.project.config = str(generated)
        with patch("inspect_rewind.sandbox.auto_compose_dir", return_value=self.root):
            with self.assertRaises(ValueError):
                self.validate({"networks": {"default": {"external": True}}})
        self.assertFalse(generated.exists())
        self.assertTrue(self.config.exists())

    def test_compose_failure_stops_startup(self) -> None:
        command = SimpleNamespace(success=False, stdout="", stderr="invalid Compose file")
        with patch("inspect_rewind.sandbox.ComposeProject.create", new_callable=AsyncMock, return_value=self.project), \
                patch("inspect_rewind.sandbox.compose_command", new_callable=AsyncMock, return_value=command), \
                patch("inspect_rewind.sandbox.DockerSandboxEnvironment.sample_init", new_callable=AsyncMock) as startup:
            with self.assertRaisesRegex(RuntimeError, "invalid Compose file"):
                anyio.run(RewindSandboxEnvironment.sample_init, "task", str(self.config), {})
        startup.assert_not_called()


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(docker_available(), "requires Docker")
class DockerInspectIsolation(unittest.TestCase):
    def test_shared_volume_and_network_rejection_preserves_live_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose = root / "compose.yaml"
            compose.write_text("""services:
  default:
    image: debian:bookworm-slim
    command: [sleep, infinity]
    volumes: [data:/data]
volumes:
  data:
""")
            with Environment.from_compose(compose, capture={"default": ["/data"]}) as parent:
                self.assertEqual(parent.exec("default", "printf original > /data/sentinel")[0], 0)
                shared = root / "shared.yaml"
                for external in (False, True):
                    config = yaml.safe_load(compose.read_text())
                    config["volumes"]["data"] = {"name": f"{parent.project}_data", "external": external}
                    config["services"]["default"]["command"] = [
                        "sh", "-c", "printf modified > /data/sentinel; sleep infinity",
                    ]
                    shared.write_text(yaml.safe_dump(config))
                    with self.subTest(external=external), self.assertRaisesRegex(ValueError, "shared across projects"):
                        anyio.run(RewindSandboxEnvironment.sample_init, "isolation", str(shared), {})
                    self.assertEqual(parent.exec("default", "cat /data/sentinel"), (0, "original"))

                config = yaml.safe_load(compose.read_text())
                config["networks"] = {"default": {"name": f"{parent.project}_default", "external": True}}
                shared.write_text(yaml.safe_dump(config))
                with self.assertRaisesRegex(ValueError, "networks 'default'.*shared across projects"):
                    anyio.run(RewindSandboxEnvironment.sample_init, "isolation", str(shared), {})
                self.assertEqual(parent.exec("default", "cat /data/sentinel"), (0, "original"))

    def test_bind_policy_uses_resolved_compose_short_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            sentinel = inputs / "sentinel"
            sentinel.write_text("original")
            config = root / "compose.yaml"
            service = {"image": "debian:bookworm-slim",
                       "command": ["sh", "-c", "printf modified > /input/sentinel; sleep infinity"]}
            for mode in ("rw", "ro"):
                service["volumes"] = [f"{inputs}:/input:{mode}"]
                config.write_text(yaml.safe_dump({"services": {"default": service}}))
                if mode == "rw":
                    with self.assertRaisesRegex(ValueError, "writable bind mount"):
                        anyio.run(RewindSandboxEnvironment.sample_init, "isolation", str(config), {})
                else:
                    anyio.run(_validate_isolation, "isolation", str(config), {})
                self.assertEqual(sentinel.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
