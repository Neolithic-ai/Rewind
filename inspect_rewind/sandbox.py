"""The `rewind` sandbox: Inspect's docker (compose) sandbox plus a snapshot
after every tool call.

`RewindSandboxEnvironment` subclasses `DockerSandboxEnvironment`, so it takes
the same `compose.yaml`/`Dockerfile` config and is docker-compatible: an
unmodified task declaring `sandbox=("docker", "compose.yaml")` runs on it with
`inspect eval task.py --sandbox rewind` (Inspect forwards the task's compose
file to a docker-compatible override, see README.md).

Snapshots reuse the `rewind/` engine (`rewind.env`): pause every service ->
`docker cp` each captured path out as a tar -> unpause, with a content hash per
tar. One snapshot is taken after every `exec`/`write_file` that runs inside a
tool call; a tool call that touches the sandbox several times keeps only its
last snapshot, so step N is "the sandbox after the Nth tool call that used it".
Each snapshot is announced with an info event (`source="rewind"`) inside the
tool call's span, so the `.eval` log itself says where step N is stored.

What is captured, per service (first match wins):
  1. `INSPECT_REWIND_CAPTURE="svc:/path,svc:/other"` (services listed here),
  2. `x-rewind: {capture: [/path, ...]}` on the service in compose.yaml,
  3. the default: every named volume the service mounts, plus, for the
     default service, its working directory (unless `/`), `/root` and `/tmp`.
Paths that don't exist when the sample starts are skipped.

Snapshots go to `$INSPECT_REWIND_DIR` (default `./.rewind`), one directory per
sample: `<log file stem>/<sample id>-e<epoch>/step-NNN/`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
from inspect_ai.event import ToolEvent
from inspect_ai.log import transcript
from inspect_ai.util import SandboxEnvironment, sandboxenv
from inspect_ai.util._sandbox.docker.compose import compose_command, compose_services
from inspect_ai.util._sandbox.docker.config import auto_compose_dir
from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment, resolve_config_environment
from inspect_ai.util._sandbox.docker.util import ComposeProject, task_project_name
from typing_extensions import override

from rewind.env import Env, _run, snapshot, tar_digest, tar_name
from rewind.isolation import validate_compose_isolation

SOURCE = "rewind"  # `source` of every info event this package writes
DEFAULT_PATHS = ["/root", "/tmp"]  # extra default captures for the default service


def snapshot_root() -> Path:
    return Path(os.environ.get("INSPECT_REWIND_DIR", ".rewind")).resolve()


def _capture_override() -> dict[str, list[str]]:
    spec: dict[str, list[str]] = {}
    for item in filter(None, os.environ.get("INSPECT_REWIND_CAPTURE", "").split(",")):
        svc, _, path = item.strip().partition(":")
        if not path.startswith("/"):
            raise ValueError(f"INSPECT_REWIND_CAPTURE item must be svc:/path, got {item!r}")
        spec.setdefault(svc, []).append(path)
    return spec


def _outermost(paths: list[str]) -> list[str]:
    """Dedupe, and drop paths nested inside another captured path."""
    norm = sorted({p.rstrip("/") or "/" for p in paths})
    return [p for p in norm if not any(p != q and p.startswith(q.rstrip("/") + "/") for q in norm)]


class ConcurrentToolCallsError(RuntimeError):
    """Concurrent tools cannot produce isolated per-call checkpoints."""


def current_tool_call_id() -> str | None:
    """The sole executing tool's id, or None outside tool calls.

    Inspect can construct ToolEvents before entering their tool spans, so
    their span IDs cannot reliably identify concurrent calls. Even with
    correct IDs, overlapping mutations have no isolated per-call state.
    Refuse the ambiguity rather than record a misleading checkpoint.
    """
    pending = [e for e in transcript().pending_events if isinstance(e, ToolEvent)]
    if not pending:
        return None
    if len(pending) > 1:
        raise ConcurrentToolCallsError(
            "rewind: concurrent tool execution is unsupported for per-call "
            "checkpoints. Run tools serially (for example, @tool(parallel=False)) "
            "or request one tool call per model turn (parallel_tool_calls=False). "
            f"Overlapping calls: {', '.join(e.id for e in pending)}."
        )
    return pending[0].id


@dataclass
class Recorder:
    """Per-sample snapshot state, shared by the sample's service environments."""

    env: Env  # the `rewind.env` engine, pointed at Inspect's compose project
    root: Path
    capture: dict[str, list[str]]
    step: int = 0
    last_call: str | None = None
    steps: dict[int, dict] = field(default_factory=dict)
    lock: anyio.Lock = field(default_factory=anyio.Lock)
    failure: str | None = None

    def index(self) -> None:
        (self.root / "index.json").write_text(json.dumps(
            {"capture": self.capture, "steps": {str(k): v for k, v in sorted(self.steps.items())}},
            indent=2))

    def before_op(self) -> str | None:
        if self.failure is not None:
            raise ConcurrentToolCallsError(self.failure)
        try:
            return current_tool_call_id()
        except ConcurrentToolCallsError as exc:
            # Keep an already-running sibling from publishing a checkpoint
            # after the other call has failed and left the pending list.
            self.failure = str(exc)
            raise

    async def after_op(self) -> None:
        async with self.lock:
            call = self.before_op()
            if call is None:
                return  # setup, scoring, ...: not an agent step
            step = self.step + (call != self.last_call)
            ckpt = self.root / f"step-{step:03d}"
            with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=self.root) as tmp:
                pending = Path(tmp) / "checkpoint"
                meta = await anyio.to_thread.run_sync(lambda: snapshot(
                    self.env, pending, {"step": step, "tool_call_id": call}))
                self.before_op()  # another tool may have started during the snapshot
                if ckpt.exists():
                    shutil.rmtree(ckpt)  # same tool call touched the sandbox again
                pending.rename(ckpt)
            self.step, self.last_call = step, call
            entry = {"step": self.step, "tool_call_id": call, "snapshot_dir": str(ckpt),
                     "content_sha256": meta["content_sha256"],
                     "save_seconds": meta["save_seconds"], "disk_bytes": meta["disk_bytes"]}
            self.steps[self.step] = entry
            self.index()
            transcript().info({"rewind": "snapshot", **entry}, source=SOURCE)


def _engine(project: Any, capture: dict[str, list[str]]) -> Env:
    """A `rewind.env.Env` for an Inspect compose project (same `-p` and `-f`)."""
    services = {svc: SimpleNamespace(name=svc, capture=paths, pre_snapshot=None, ready=None)
                for svc, paths in capture.items()}
    scenario = SimpleNamespace(compose_file=project.config, services=services, root=Path.cwd())
    return Env(scenario=scenario, project=project.name)  # type: ignore[arg-type]


async def _validate_isolation(task_name: str, config: Any, metadata: dict[str, str]) -> None:
    """Resolve exactly as Inspect does, before it starts or registers resources."""
    resolved = resolve_config_environment(config, metadata)
    project = await ComposeProject.create(
        name=task_project_name(task_name), config=config,
        env=resolved.env if resolved is not None else {},
    )
    # ComposeProject.create can materialize an inline ComposeConfig or Dockerfile.
    # Remove only the file generated for this validation, never the user's config
    # and never Docker resources (which may be the very shared resources we reject).
    generated_config = auto_compose_dir() / f"{project.name}.yaml"
    try:
        result = await compose_command(["config", "--format", "json"], project=project, timeout=300)
        if not result.success:
            raise RuntimeError(f"Error reading Docker config for Rewind isolation: {result.stderr}")
        validate_compose_isolation(json.loads(result.stdout), project.name)
    finally:
        if project.config and Path(project.config).resolve() == generated_config.resolve():
            generated_config.unlink(missing_ok=True)


@sandboxenv(name="rewind")
class RewindSandboxEnvironment(DockerSandboxEnvironment):
    """Docker compose sandbox that snapshots after every tool call."""

    _rewind: Recorder

    @override
    @classmethod
    async def sample_init(cls, task_name: str, config: Any,
                          metadata: dict[str, str]) -> dict[str, SandboxEnvironment]:
        await _validate_isolation(task_name, config, metadata)
        docker_envs = await super().sample_init(task_name, config, metadata)
        first = next(iter(docker_envs.values())).as_type(DockerSandboxEnvironment)
        project = first._project
        services = await compose_services(project)
        override_spec = _capture_override()
        capture: dict[str, list[str]] = {}
        for i, (svc, env) in enumerate(docker_envs.items()):
            info = services.get(svc, {})
            if svc in override_spec:
                paths = override_spec[svc]
            elif isinstance(info.get("x-rewind"), dict) and "capture" in info["x-rewind"]:
                paths = list(info["x-rewind"]["capture"] or [])
            else:
                paths = [v["target"] for v in info.get("volumes") or []
                         if isinstance(v, dict) and v.get("type") == "volume"]
                if i == 0:  # sample_init puts the default service first
                    wd = env.as_type(DockerSandboxEnvironment)._working_dir
                    paths += ([wd] if wd != "/" else []) + DEFAULT_PATHS
            existing = []
            for p in _outermost(paths):
                if p == "/":
                    raise ValueError(f"rewind: refusing to capture / in service {svc!r}")
                if (await env.exec(["test", "-e", p])).success:
                    existing.append(p)
            if existing:
                capture[svc] = existing

        from inspect_ai.log._samples import sample_active

        active = sample_active()
        stem = Path(active.log_location).stem if active and active.log_location else "eval"
        sid = f"{active.sample.id}-e{active.epoch}" if active else "sample"
        root = snapshot_root() / stem / sid
        root.mkdir(parents=True, exist_ok=True)
        rec = Recorder(env=_engine(project, capture), root=root, capture=capture)
        rec.index()
        transcript().info({"rewind": "capture", "capture": capture, "root": str(root)},
                          source=SOURCE)

        envs: dict[str, SandboxEnvironment] = {}
        for svc, env in docker_envs.items():
            d = env.as_type(DockerSandboxEnvironment)
            new = cls(d._service, d._project, d._working_dir)
            new._rewind = rec
            envs[svc] = new
        return envs

    @override
    async def exec(self, cmd: list[str], *args: Any, **kwargs: Any):  # type: ignore[override]
        self._rewind.before_op()
        result = await super().exec(cmd, *args, **kwargs)
        await self._rewind.after_op()
        return result

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        self._rewind.before_op()
        await super().write_file(file, contents)
        await self._rewind.after_op()

    # --- branching ------------------------------------------------------------

    async def restore(self, ckpt: Path, step: int, inherited: dict[int, dict]) -> dict:
        """Put snapshot `ckpt` (taken at `step` of a parent) into this sample's
        running project and verify it.

        Every captured path is emptied, the project is stopped, the snapshot's
        tars are copied in, each path is copied back out and its content hash
        compared with the one recorded at snapshot time, then the project is
        started again (processes restart via their entrypoints). Later
        snapshots of this sample continue the parent's numbering at step+1.
        Returns {"ok", "paths": {tar: {"expected", "restored", "ok"}}, "seconds"}.
        """
        rec = self._rewind
        meta = json.loads((ckpt / "meta.json").read_text())
        expected: dict[str, str] = meta["content_sha256"]
        tars = {tar_name(svc, p): (svc, p) for svc, paths in rec.capture.items() for p in paths}
        if set(tars) != set(expected):
            raise RuntimeError(f"snapshot {ckpt} holds {sorted(expected)}, but this sandbox "
                               f"captures {sorted(tars)}: different compose file or capture?")

        def run() -> dict:
            t0 = time.time()
            env = rec.env
            for name, (svc, path) in tars.items():
                _run(["docker", "exec", env.container_id(svc), "sh", "-c",
                      f'find "{path}" -mindepth 1 -delete'], check=False)
            _run(env.compose("stop", "-t", "2"))
            paths = {}
            with tempfile.TemporaryDirectory() as tmp:
                for name, (svc, path) in tars.items():
                    env.cp_in(svc, os.path.dirname(path.rstrip("/")) or "/", ckpt / name)
                    env.cp_out(svc, path, Path(tmp) / name)
                    got = tar_digest(Path(tmp) / name)
                    paths[name] = {"expected": expected[name], "restored": got,
                                   "ok": got == expected[name]}
            env.start()
            _wait_running(env)
            return {"ok": all(p["ok"] for p in paths.values()), "paths": paths,
                    "seconds": round(time.time() - t0, 3)}

        result = await anyio.to_thread.run_sync(run)
        rec.step, rec.last_call = step, None
        rec.steps = {k: v for k, v in inherited.items() if k <= step}
        rec.index()
        return result


def _wait_running(env: Env, timeout: float = 120) -> None:
    """Wait until every service is running (and healthy, if it has a healthcheck)."""
    deadline = time.time() + timeout
    while True:
        out = _run(env.compose("ps", "-a", "--format", "json")).stdout.decode()
        out = out.strip()
        rows = json.loads(out) if out.startswith("[") else \
            [json.loads(line) for line in out.splitlines() if line.strip()]
        if rows and all(r.get("State") == "running" and r.get("Health") in ("", None, "healthy")
                        for r in rows):
            return
        if time.time() > deadline:
            raise RuntimeError(f"services not running after restore: {rows}")
        time.sleep(0.3)


__all__ = ["RewindSandboxEnvironment", "current_tool_call_id", "snapshot_root"]
