"""Docker filesystem checkpoints for any agent, with explicit resource ownership.

The caller supplies its own JSON transcript and chooses when to checkpoint.
Processes restart through Compose entrypoints; memory, sessions and network
addresses are not captured by this backend.
"""

from __future__ import annotations

import json
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Protocol

from .env import Env, _run, restore, snapshot, tar_digest, tar_name
from .isolation import validate_compose_isolation
from .scenario import Service


CAPABILITIES = {
    "filesystem": True,
    "process_memory": False,
    "sessions": False,
    "network_addresses": False,
}
SCHEMA_VERSION = 1


class Lifecycle(Protocol):
    """Caller-supplied application state adapter, never loaded from a checkpoint.

    Use a stable, versioned name. before_checkpoint must quiesce writers and
    flush state into captured paths. after_checkpoint must safely resume even
    after a partial preparation failure. after_restore must hydrate, validate,
    and resume services, raising on failure. Services must start inert until
    this hook runs; filesystem verification alone cannot enforce that contract.
    Hooks run synchronously and should use bounded timeouts.
    """

    name: str

    def before_checkpoint(self, environment: Environment) -> None: ...
    def after_checkpoint(self, environment: Environment) -> None: ...
    def after_restore(self, environment: Environment, checkpoint: Checkpoint) -> None: ...


def _require_lifecycle(name: str | None, lifecycle: Lifecycle | None) -> None:
    if name is not None and (lifecycle is None or lifecycle.name != name):
        raise ValueError(f"checkpoint requires lifecycle adapter {name!r}; supply it explicitly")


def _capture_paths(capture: dict[str, list[str]]) -> dict[str, list[str]]:
    if not isinstance(capture, dict) or not capture:
        raise ValueError("capture must map at least one service to container paths")
    result: dict[str, list[str]] = {}
    archives: set[str] = set()
    for service, paths in capture.items():
        if not isinstance(service, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", service):
            raise ValueError(f"invalid capture service: {service!r}")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"capture for {service!r} must be a nonempty list of absolute paths")
        normalized: set[str] = set()
        for path in paths:
            if (not isinstance(path, str) or not path.startswith("/") or "\0" in path
                    or ".." in PurePosixPath(path).parts):
                raise ValueError(f"capture path must be absolute without '..': {path!r}")
            path = "/" + str(PurePosixPath(path)).lstrip("/")
            if path == "/":
                raise ValueError("capturing the container root / is unsupported; select state paths")
            normalized.add(path)
        # Capturing a parent already includes its children.
        selected = [p for p in sorted(normalized)
                    if not any(p != q and p.startswith(q + "/") for q in normalized)]
        for path in selected:
            archive = tar_name(service, path)
            if archive in archives:
                raise ValueError(f"capture paths produce the same archive name: {archive}")
            archives.add(archive)
        result[service] = selected
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read Rewind metadata at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Rewind metadata at {path} must be a JSON object")
    return value


def _check_format(data: dict, kind: str) -> None:
    if data.get("kind") != kind or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected {kind} metadata with schema_version={SCHEMA_VERSION}")
    if data.get("backend") != "filesystem":
        raise ValueError(f"unsupported checkpoint backend: {data.get('backend')!r}")


@dataclass(frozen=True)
class Checkpoint:
    """An on-disk snapshot and caller-owned JSON metadata/messages."""

    path: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> Checkpoint:
        directory = Path(path).expanduser().resolve()
        manifest = _read_json(directory / "meta.json")
        _check_format(manifest, "rewind.checkpoint")
        _capture_paths(manifest.get("capture"))
        if not isinstance(manifest.get("compose_file"), str):
            raise ValueError("checkpoint metadata is missing compose_file")
        return cls(directory, manifest)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.manifest.get("metadata") or {}

    def export(self, output: str | Path) -> Path:
        """Bundle captured files, messages, Compose configuration and declared inputs."""
        from .bundle import export_checkpoint
        return export_checkpoint(self, output)

    @classmethod
    def import_bundle(cls, archive: str | Path, output: str | Path) -> Checkpoint:
        """Verify and import a bundle without starting Docker or an agent."""
        from .bundle import import_checkpoint
        return import_checkpoint(archive, output)

    @property
    def messages(self) -> Any:
        return self.manifest.get("messages")

    @property
    def restore_notice(self) -> str:
        """Optional context for the caller to give its agent after branching."""
        return ("<rewind_notice>Captured filesystem paths were restored and services restarted. "
                "Process memory, background processes, and live shell or network sessions were "
                "not restored. Network addresses may have changed. Verify endpoints and "
                "reconnect as needed, then continue from the saved conversation.</rewind_notice>")

    def verify(self) -> dict[str, str]:
        """Validate every archive before creating or changing Docker resources."""
        capture = _capture_paths(self.manifest.get("capture"))
        names = {tar_name(service, path) for service, paths in capture.items() for path in paths}
        expected = self.manifest.get("content_sha256")
        if not isinstance(expected, dict) or set(expected) != names:
            raise ValueError("checkpoint archive hashes do not match its capture paths")
        for name in sorted(names):
            archive = self.path / name
            if not archive.is_file():
                raise FileNotFoundError(f"checkpoint archive is missing: {archive}")
            try:
                actual = tar_digest(archive)
            except Exception as exc:
                raise ValueError(f"cannot read checkpoint archive {archive}: {exc}") from exc
            if actual != expected[name]:
                raise ValueError(f"checkpoint archive failed hash verification: {archive}")
        return dict(expected)

    def branch(self, *, project: str | None = None,
               compose_file: str | Path | None = None,
               lifecycle: Lifecycle | None = None) -> Environment:
        """Restore into a new project; failed restores remove only that new project."""
        _require_lifecycle(self.manifest.get("lifecycle"), lifecycle)
        self.verify()
        environment = Environment.from_compose(
            compose_file or self.manifest["compose_file"],
            capture=self.manifest["capture"], project=project, lifecycle=lifecycle)
        if environment.project == self.manifest.get("project"):
            raise ValueError("a branch needs a different Compose project from its parent")
        environment._prepare()
        environment._owned = True
        try:
            seconds, report = restore(environment._engine, self.path, verify=True)
            environment._started = True
            environment.restore_report = {
                **report, "seconds": seconds, "checkpoint": str(self.path),
                "capabilities": dict(CAPABILITIES),
                "restore_mode": "copy_into_fresh_container",
            }
            if lifecycle is not None:
                lifecycle.after_restore(environment, self)
                environment.restore_report["lifecycle"] = lifecycle.name
            return environment
        except BaseException as exc:
            environment._cleanup_failed(exc)
            raise


class Environment:
    """An isolated Docker Compose project, independent of any agent framework."""

    def __init__(self, compose_file: Path, capture: dict[str, list[str]], project: str,
                 lifecycle: Lifecycle | None = None) -> None:
        if lifecycle is not None and (not isinstance(lifecycle.name, str) or not lifecycle.name.strip()):
            raise ValueError("lifecycle adapter needs a nonempty name")
        self.lifecycle = lifecycle
        self.compose_file = compose_file
        self.capture = capture
        self.project = project
        self._started = False
        self._owned = False
        self.restore_report: dict[str, Any] | None = None
        config = SimpleNamespace(
            root=compose_file.parent, compose_file=compose_file,
            services={service: Service(name=service, capture=paths)
                      for service, paths in capture.items()})
        self._engine = Env(config, project)  # type: ignore[arg-type]

    @classmethod
    def from_compose(cls, compose_file: str | Path, *, capture: dict[str, list[str]],
                     project: str | None = None, lifecycle: Lifecycle | None = None) -> Environment:
        """Configure an environment; no containers are created until start()."""
        path = Path(compose_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Compose file not found: {path}")
        name = project or f"rewind_{uuid.uuid4().hex[:12]}"
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
            raise ValueError("project must start with a lowercase letter or digit and use a-z, 0-9, _ or -")
        return cls(path, _capture_paths(capture), name, lifecycle)

    @property
    def started(self) -> bool:
        return self._started

    def _prepare(self) -> None:
        """Reject resource sharing that would defeat branch isolation."""
        config = json.loads(_run(self._engine.compose("config", "--format", "json")).stdout)
        services = config.get("services") or {}
        missing = set(self.capture) - set(services)
        if missing:
            raise ValueError(f"capture names unknown Compose services: {sorted(missing)}")
        validate_compose_isolation(config, self.project)
        # Never adopt an existing project: failure cleanup must only remove our resources.
        for command in (["docker", "ps", "-aq"], ["docker", "volume", "ls", "-q"],
                        ["docker", "network", "ls", "-q"]):
            found = _run([*command, "--filter", f"label=com.docker.compose.project={self.project}"])
            if found.stdout.strip():
                raise ValueError(f"Compose project {self.project!r} already has resources; choose a new project name")

    def start(self) -> Environment:
        if self._started:
            return self
        self._prepare()
        self._owned = True
        try:
            self._engine.up()
            self._started = True
            return self
        except BaseException as exc:
            self._cleanup_failed(exc)
            raise

    def exec(self, service: str, command: str | list[str], **kwargs: Any) -> tuple[int, str]:
        """Run argv directly or a string through sh; return (exit code, combined output)."""
        if not self._started:
            raise RuntimeError("environment is not started; call start() first")
        if isinstance(command, str):
            command = ["sh", "-c", command]
        if not command or not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
            raise ValueError("command must be a shell string or a nonempty list of arguments")
        return self._engine.exec(service, command, **kwargs)

    def checkpoint(self, output_dir: str | Path, *, metadata: dict[str, Any] | None = None,
                   messages: Any = None) -> Checkpoint:
        """Pause services and snapshot captured paths; calls must be quiescent."""
        if not self._started:
            raise RuntimeError("environment is not started; call start() first")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        details = {
            "kind": "rewind.checkpoint", "schema_version": SCHEMA_VERSION,
            "backend": "filesystem", "compose_file": str(self.compose_file),
            "project": self.project, "capture": self.capture,
            "metadata": metadata or {}, "messages": messages,
            "capabilities": dict(CAPABILITIES),
            "restore_mode": "copy_into_fresh_container",
        }
        if self.lifecycle is not None:
            details["lifecycle"] = self.lifecycle.name
        try:
            details = json.loads(json.dumps(details))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint metadata and messages must be JSON-serializable: {exc}") from exc
        output = Path(output_dir).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"checkpoint destination already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
            staged = Path(tmp) / "checkpoint"
            try:
                if self.lifecycle is not None:
                    self.lifecycle.before_checkpoint(self)
                for service, paths in self.capture.items():
                    for path in paths:
                        exists = _run(["docker", "exec", self._engine.container_id(service),
                                       "test", "-e", path], check=False)
                        if exists.returncode:
                            raise FileNotFoundError(f"capture path does not exist: {service}:{path}")
                manifest = snapshot(self._engine, staged, details)
            except BaseException as exc:
                if self.lifecycle is not None:
                    try:
                        self.lifecycle.after_checkpoint(self)
                    except BaseException as resume_error:
                        exc.add_note(f"resuming after checkpoint failure also failed: {resume_error}")
                raise
            else:
                if self.lifecycle is not None:
                    self.lifecycle.after_checkpoint(self)
            staged.rename(output)
        return Checkpoint(output, manifest)

    @classmethod
    def branch(cls, checkpoint: Checkpoint | str | Path, *, project: str | None = None,
               compose_file: str | Path | None = None,
               lifecycle: Lifecycle | None = None) -> Environment:
        saved = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint.load(checkpoint)
        return saved.branch(project=project, compose_file=compose_file, lifecycle=lifecycle)

    def save(self, path: str | Path) -> Path:
        """Save a handle to these containers; this is not a filesystem checkpoint."""
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "kind": "rewind.environment", "schema_version": SCHEMA_VERSION,
            "backend": "filesystem", "compose_file": str(self.compose_file),
            "project": self.project, "capture": self.capture, "started": self._started,
            "owned": self._owned,
            "restore_report": self.restore_report,
            "lifecycle": self.lifecycle.name if self.lifecycle is not None else None,
        }
        temporary = target.with_name(f".{target.name}-{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(data, indent=2))
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def load(cls, path: str | Path, *, compose_file: str | Path | None = None,
               lifecycle: Lifecycle | None = None) -> Environment:
        data = _read_json(Path(path).expanduser().resolve())
        _check_format(data, "rewind.environment")
        if (not isinstance(data.get("compose_file"), str)
                or not isinstance(data.get("project"), str)
                or not isinstance(data.get("started"), bool)
                or not isinstance(data.get("owned", data.get("started")), bool)):
            raise ValueError("environment state needs compose_file, project, and a boolean started flag")
        _require_lifecycle(data.get("lifecycle"), lifecycle)
        environment = cls.from_compose(compose_file or data["compose_file"],
                                       capture=data.get("capture"), project=data["project"], lifecycle=lifecycle)
        environment._started = bool(data.get("started"))
        environment._owned = data.get("owned", environment._started)
        if environment._started and not environment._owned:
            raise ValueError("a started environment handle must own its project")
        environment.restore_report = data.get("restore_report")
        return environment

    def down(self) -> None:
        """Remove this project's containers, network and named volumes."""
        if not self._owned:
            return
        self._engine.down(check=True)
        self._started = False
        self._owned = False

    def _cleanup_failed(self, error: BaseException) -> None:
        try:
            self.down()
        except Exception as cleanup_error:
            error.add_note(f"cleanup of project {self.project!r} also failed: {cleanup_error}")

    def __enter__(self) -> Environment:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        if exc[1] is not None:
            self._cleanup_failed(exc[1])
        else:
            self.down()
