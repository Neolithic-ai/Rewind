"""Environment = one `docker compose` project; checkpoint = tars of captured paths.

Capture and restore operate on the paths declared by the calling experiment:

- Two environments are fully isolated because each is its own compose project
  (`-p <project>`): separate containers, network and named volumes.
- `snapshot` runs every service's optional pre-snapshot hook (e.g. a SQLite WAL
  checkpoint), pauses *all* services (`docker compose pause`, one consistent cut
  across containers), pulls each captured path out as a tar with `docker cp`,
  then unpauses, and records a content hash of each tar.
- `restore` creates a fresh project without starting it, pushes every tar back
  with `docker cp` (so data is in place before any service boots), copies each
  path out again and checks its content hash against the one recorded at
  snapshot time, then starts the project and waits for the readiness checks.
  Running processes are not restored; services come back via their normal
  entrypoints.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .scenario import Scenario

# Resolve once so a run and its subprocesses agree on the artifact location.
# Installed packages must never write into their own site-packages directory.
RUNS_DIR = Path(os.environ.get("REWIND_RUNS_DIR") or "runs").expanduser().resolve()


def _run(argv: list[str], *, input_bytes: bytes | None = None, check: bool = True,
         stdout_path: Path | None = None, timeout: float | None = None,
         env: dict[str, str] | None = None
         ) -> subprocess.CompletedProcess:
    stdout = open(stdout_path, "wb") if stdout_path else subprocess.PIPE
    try:
        proc = subprocess.run(argv, input=input_bytes, stdout=stdout,
                              stderr=subprocess.PIPE, check=False, timeout=timeout, env=env)
    finally:
        if stdout_path:
            stdout.close()
    if check and proc.returncode != 0:
        err = proc.stderr.decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(argv)}\n{err}")
    return proc


def exit_on_sigterm() -> None:
    """Turn SIGTERM (e.g. a cancelled CI job) into a normal exit, so `finally`
    and `except` blocks still tear environments down."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))


def tar_name(service: str, path: str) -> str:
    return f"{service}__{path.strip('/').replace('/', '_') or 'root'}.tar"


def tar_digest(tar: Path) -> str:
    """Content hash of a `docker cp` tar: every entry's path, type, mode, owner,
    link target and bytes (not mtimes), in path order."""
    h = hashlib.sha256()
    with tarfile.open(tar) as t:
        for m in sorted(t.getmembers(), key=lambda m: m.name):
            h.update(f"{m.name}\0{m.type.decode()}\0{m.mode:o}\0{m.uid}:{m.gid}\0"
                     f"{m.linkname}\0{m.size if m.isfile() else 0}\0".encode())
            if m.isfile():
                f = t.extractfile(m)
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


@dataclass
class Env:
    scenario: Scenario
    project: str
    _cids: dict[str, str] = field(default_factory=dict)

    # --- compose lifecycle -------------------------------------------------
    def compose(self, *args: str) -> list[str]:
        return ["docker", "compose", "-p", self.project, "-f",
                str(self.scenario.compose_file), *args]

    def up(self) -> None:
        _run(self.compose("up", "-d"))
        self._cids.clear()

    def create(self) -> None:
        _run(self.compose("create"))
        self._cids.clear()

    def start(self) -> None:
        _run(self.compose("start"))

    def down(self, *, check: bool = False) -> None:
        _run(self.compose("down", "-v", "-t", "2"), check=check)

    def pause(self) -> None:
        _run(self.compose("pause"))

    def unpause(self) -> None:
        _run(self.compose("unpause"), check=False)

    def container_id(self, service: str) -> str:
        if service not in self._cids:
            cid = _run(self.compose("ps", "-aq", service)).stdout.decode().strip()
            if not cid:
                raise RuntimeError(f"no container for {service!r} in {self.project}")
            self._cids[service] = cid.splitlines()[0]
        return self._cids[service]

    # --- exec / copy -------------------------------------------------------
    def exec(self, service: str, cmd: str | list[str], *, user: str | None = None,
             env: dict[str, str] | None = None, workdir: str | None = None,
             timeout: float | None = None) -> tuple[int, str]:
        """Run argv directly or `bash -c <cmd>` for strings; return (rc, stdout+stderr)."""
        argv = ["docker", "exec"]
        if user:
            argv += ["-u", user]
        if workdir:
            argv += ["-w", workdir]
        for k, v in (env or {}).items():
            argv += ["-e", f"{k}={v}"]
        argv += [self.container_id(service), *(["bash", "-c", cmd] if isinstance(cmd, str) else cmd)]
        try:
            proc = _run(argv, check=False, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, f"(timed out after {timeout}s)"
        out = (proc.stdout or b"").decode(errors="replace") + \
              (proc.stderr or b"").decode(errors="replace")
        return proc.returncode, out

    def cp_out(self, service: str, path: str, dest: Path) -> int:
        _run(["docker", "cp", f"{self.container_id(service)}:{path}", "-"], stdout_path=dest)
        return dest.stat().st_size

    def cp_in(self, service: str, dest_parent: str, tar: Path) -> None:
        _run(["docker", "cp", "-", f"{self.container_id(service)}:{dest_parent}"],
             input_bytes=tar.read_bytes())

    def cp_dir_in(self, service: str, src: Path, dest: str) -> None:
        _run(["docker", "cp", str(src), f"{self.container_id(service)}:{dest}"])

    # --- readiness ---------------------------------------------------------
    def wait_ready(self, timeout: float = 120) -> None:
        for svc in self.scenario.services.values():
            if not svc.ready:
                continue
            deadline = time.time() + timeout
            while True:
                rc, _ = self.exec(svc.name, svc.ready, timeout=10)
                if rc == 0:
                    break
                if time.time() > deadline:
                    raise RuntimeError(f"{svc.name} not ready after {timeout}s")
                time.sleep(0.5)


# --- seed --------------------------------------------------------------------

def seed(env: Env) -> dict[str, str]:
    """Run the scenario's seed steps on a fresh env; return all variables."""
    sc = env.scenario
    variables = dict(sc.vars)
    for step in sc.seed:
        if step.copy:
            env.cp_dir_in(step.service, sc.root / step.copy, step.to)
            continue
        cmd = (sc.root / step.script).read_text() if step.script else step.run
        rc, out = env.exec(step.service, cmd, user=step.user, env=variables)
        if rc != 0:
            raise RuntimeError(f"seed step {step.name!r} failed ({rc}):\n{out}")
        if step.save_as:
            variables[step.save_as] = out.strip().splitlines()[-1].strip()
    return variables


# --- checkpoint / restore ----------------------------------------------------

def snapshot(env: Env, ckpt_dir: Path, meta: dict) -> dict:
    """Hooks -> pause all -> docker cp every captured path -> unpause."""
    sc = env.scenario
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for svc in sc.services.values():
        if svc.pre_snapshot:
            rc, out = env.exec(svc.name, svc.pre_snapshot.source(sc.root),
                               user=svc.pre_snapshot.user)
            if rc != 0:
                raise RuntimeError(f"pre-snapshot hook for {svc.name} failed: {out}")
    t_hooks = time.time()
    sizes: dict[str, int] = {}
    env.pause()
    try:
        for svc in sc.services.values():
            for path in svc.capture:
                name = tar_name(svc.name, path)
                sizes[name] = env.cp_out(svc.name, path, ckpt_dir / name)
    finally:
        env.unpause()
    t1 = time.time()
    meta = {
        **meta,
        "content_sha256": {name: tar_digest(ckpt_dir / name) for name in sizes},
        "hook_seconds": round(t_hooks - t0, 3),
        "paused_seconds": round(t1 - t_hooks, 3),
        "save_seconds": round(t1 - t0, 3),
        "disk_bytes": sum(sizes.values()),
        "sizes": sizes,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def restore(env: Env, ckpt_dir: Path, *, verify: bool = False) -> tuple[float, dict]:
    """Restore a checkpoint into a fresh (not yet existing) env.

    Before any service starts, every restored path is copied back out and its
    content hash compared with the one recorded when the checkpoint was taken
    (`ok` is None for checkpoints that predate the hashes). Returns
    (seconds, {"ok", "paths": {tar: {"expected", "restored", "ok"}}}).
    With `verify=True`, a missing or mismatched hash raises before services start.
    """
    t0 = time.time()
    expected = json.loads((ckpt_dir / "meta.json").read_text()).get("content_sha256") or {}
    env.create()
    paths: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for svc in env.scenario.services.values():
            for path in svc.capture:
                name = tar_name(svc.name, path)
                parent = os.path.dirname(path.rstrip("/")) or "/"
                try:
                    env.cp_in(svc.name, parent, ckpt_dir / name)
                except RuntimeError as exc:
                    if verify:
                        raise RuntimeError(
                            f"could not restore {svc.name}:{path}: {exc}\n"
                            f"The captured path's parent {parent!r} must already exist in the fresh "
                            "image. Capture an existing parent directory or mount the state "
                            "directory as a dedicated named volume.") from exc
                    raise
                env.cp_out(svc.name, path, Path(tmp) / name)
                got, want = tar_digest(Path(tmp) / name), expected.get(name)
                paths[name] = {"expected": want, "restored": got,
                               "ok": None if want is None else got == want}
    oks = [p["ok"] for p in paths.values()]
    content = {"ok": None if None in oks else all(oks), "paths": paths}
    if verify and content["ok"] is not True:
        bad = [name for name, result in paths.items() if result["ok"] is not True]
        raise RuntimeError(
            f"restored filesystem does not match checkpoint: {bad}. "
            "Copy-based restore cannot remove files already present in a fresh image; this can "
            "happen when the checkpoint deleted image-provided files. Use state directories "
            "backed by initially empty dedicated named volumes.")
    env.start()
    env.wait_ready()
    return round(time.time() - t0, 3), content


def build_images(scenario: Scenario) -> None:
    if scenario.build:
        _run(["docker", "compose", "-p", "rewind_build", "-f", str(scenario.compose_file),
              "build", *scenario.build])
