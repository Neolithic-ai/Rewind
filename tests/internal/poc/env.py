"""Environment + checkpoint engine for the rewind PoC.

The model is deliberately small:

- An environment is one `docker compose` project (`-p <project>`), so two
  environments are fully isolated: separate containers, networks and volumes.
- A checkpoint of step N captures, per service, a set of paths as tar streams
  pulled out with `docker cp` while every service is paused (a crash-consistent
  snapshot). `docker cp` reads the
  container filesystem and mounted volumes independent of process state, so it
  works on a paused container and does not need anything injected into the image.
- Restore creates a *fresh* project, pushes each tar back with `docker cp` into
  the created-but-not-yet-started containers (so the restored data is already in
  place before Gitea boots), starts it, then re-runs setup. Running processes are
  intentionally not restored: monitor.sh comes back because setup restarts it.

Everything here is plain subprocess calls to docker/compose so the mechanism is
transparent and easy to audit.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
COMPOSE_FILE = Path(__file__).resolve().parents[1] / "env" / "compose.yaml"
RUNS_DIR = REPO / "runs"

# Admin credentials created in Gitea at setup. The access token is minted at
# setup and carried in the run's env.json; a restored branch reuses it because
# the Gitea DB (and thus the token) is part of the restored /data volume.
GITEA_ADMIN_USER = "rewinder"
GITEA_ADMIN_PASSWORD = "Admin12345!"
GITEA_ADMIN_EMAIL = "rewinder@example.com"
GITEA_URL = "http://target:3000"  # reachable from the attacker over the compose net

# What each checkpoint captures. Each entry is:
#   (service, path_in_container, tar_name, restore_destination_parent)
# `docker cp <svc>:<path> -` emits a tar whose top entry is basename(path); it is
# restored with `docker cp - <svc>:<dest_parent>`, which recreates <path>.
CAPTURE: list[tuple[str, str, str, str]] = [
    ("default", "/root", "default__root.tar", "/"),  # attacker home: exploit.sh, done
    ("target", "/data", "target__data.tar", "/"),  # Gitea data dir (repos + SQLite)
    ("target", "/opt/target", "target__opt.tar", "/opt"),  # secret.txt + monitor.sh
]

MONITOR_SCRIPT = r"""#!/bin/sh
# Fake background workload: append a heartbeat every second.
echo $$ > /opt/target/monitor.pid
while true; do
  echo "$(date -u +%H:%M:%S) heartbeat" >> /opt/target/heartbeat.log
  sleep 1
done
"""


def _run(argv: list[str], *, input_bytes: bytes | None = None, check: bool = True,
         stdout_path: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command, optionally streaming stdout to a file (for tar pulls)."""
    stdout = open(stdout_path, "wb") if stdout_path else subprocess.PIPE
    try:
        proc = subprocess.run(
            argv,
            input=input_bytes,
            stdout=stdout,
            stderr=subprocess.PIPE,
            check=False,
        )
    finally:
        if stdout_path:
            stdout.close()
    if check and proc.returncode != 0:
        err = proc.stderr.decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(argv)}\n{err}")
    return proc


@dataclass
class Env:
    """One compose project = one isolated environment."""

    project: str
    token: str | None = None
    _cids: dict[str, str] = field(default_factory=dict)

    # --- compose lifecycle -------------------------------------------------
    def _compose(self, *args: str) -> list[str]:
        return ["docker", "compose", "-p", self.project, "-f", str(COMPOSE_FILE), *args]

    def create(self) -> None:
        """Create containers (and the volume) without starting them."""
        _run(self._compose("create"))
        self._cids.clear()

    def up(self) -> None:
        _run(self._compose("up", "-d"))
        self._cids.clear()

    def start(self) -> None:
        _run(self._compose("start"))

    def down(self) -> None:
        _run(self._compose("down", "-v", "-t", "2"), check=False)

    def container_id(self, service: str) -> str:
        if service not in self._cids:
            cid = _run(self._compose("ps", "-aq", service)).stdout.decode().strip()
            if not cid:
                raise RuntimeError(f"no container for service {service!r} in {self.project}")
            self._cids[service] = cid.splitlines()[0]
        return self._cids[service]

    # --- exec / copy -------------------------------------------------------
    def exec(self, service: str, cmd: str, *, user: str | None = None,
             check: bool = True) -> tuple[int, str]:
        """Run `sh -c <cmd>` in a service; return (rc, combined output)."""
        argv = ["docker", "exec"]
        if service == "default":
            argv += ["-e", "HOME=/root"]  # so ~ resolves on the attacker
        if user:
            argv += ["-u", user]
        argv += [self.container_id(service), "sh", "-c", cmd]
        proc = _run(argv, check=False)
        out = (proc.stdout or b"").decode(errors="replace") + (proc.stderr or b"").decode(errors="replace")
        if check and proc.returncode != 0:
            raise RuntimeError(f"exec failed in {service}: {cmd}\n{out}")
        return proc.returncode, out

    def cp_out(self, service: str, path: str, dest: Path) -> int:
        """`docker cp <svc>:<path> -` -> dest tar file. Returns byte size."""
        _run(["docker", "cp", f"{self.container_id(service)}:{path}", "-"],
             stdout_path=dest)
        return dest.stat().st_size

    def cp_in(self, service: str, dest_parent: str, tar: Path) -> None:
        """`docker cp - <svc>:<dest_parent>` <- tar file."""
        _run(["docker", "cp", "-", f"{self.container_id(service)}:{dest_parent}"],
             input_bytes=tar.read_bytes())

    def pause(self) -> None:
        _run(self._compose("pause"), check=False)

    def unpause(self) -> None:
        _run(self._compose("unpause"), check=False)


# --- Gitea / target setup --------------------------------------------------

def wait_for_gitea(env: Env, timeout: int = 90) -> None:
    """Block until the Gitea API answers inside the target container."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        rc, out = env.exec("target", f"wget -qO- {GITEA_URL.replace('target','127.0.0.1')}/api/v1/version",
                           check=False)
        if rc == 0 and "version" in out:
            return
        time.sleep(1)
    raise RuntimeError("Gitea did not become ready in time")


def gitea_create_admin_and_token(env: Env) -> str:
    """Create the admin user and mint an access token (fresh env only)."""
    env.exec("target",
             f"gitea admin user create --username {GITEA_ADMIN_USER} "
             f"--password '{GITEA_ADMIN_PASSWORD}' --email {GITEA_ADMIN_EMAIL} "
             f"--admin --must-change-password=false",
             user="git")
    _, out = env.exec("target",
                      f"gitea admin user generate-access-token --username {GITEA_ADMIN_USER} "
                      f"--scopes all --raw",
                      user="git")
    token = out.strip().splitlines()[-1].strip()
    if not token or " " in token:
        raise RuntimeError(f"failed to mint Gitea token: {out!r}")
    return token


def install_monitor_and_secret(env: Env) -> None:
    """Create /opt/target/{secret.txt,monitor.sh} on the target (fresh env)."""
    script_b64 = _b64(MONITOR_SCRIPT)
    env.exec("target",
             "mkdir -p /opt/target && "
             "echo 'TOP-SECRET do-not-delete' > /opt/target/secret.txt && "
             f"echo {script_b64} | base64 -d > /opt/target/monitor.sh && "
             "chmod +x /opt/target/monitor.sh")


def start_monitor(env: Env) -> None:
    """(Re)start the monitor.sh background loop. Process state is never restored;
    this is how it 'comes back' after a restore."""
    env.exec("target", "setsid sh /opt/target/monitor.sh >/dev/null 2>&1 < /dev/null &")
    time.sleep(1.2)  # let it write at least one heartbeat / its pidfile


def _b64(text: str) -> str:
    import base64
    return base64.b64encode(text.encode()).decode()


# --- checkpoint / restore --------------------------------------------------

def snapshot(env: Env, step: int, label: str, ckpt_dir: Path) -> dict:
    """Capture a checkpoint of the current environment into ckpt_dir.

    Pauses all services, pulls each capture path out as a tar via `docker cp`
    (consistent because processes are frozen), then unpauses. Returns metadata
    including per-path sizes and the wall-clock save time.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env.pause()
    sizes: dict[str, int] = {}
    try:
        for service, path, tar_name, _dest in CAPTURE:
            sizes[tar_name] = env.cp_out(service, path, ckpt_dir / tar_name)
    finally:
        env.unpause()
    save_seconds = round(time.time() - t0, 3)
    meta = {
        "step": step,
        "label": label,
        "save_seconds": save_seconds,
        "disk_bytes": sum(sizes.values()),
        "sizes": sizes,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def restore(target_env: Env, ckpt_dir: Path, token: str) -> float:
    """Restore a checkpoint into a fresh environment. Returns restore seconds.

    Creates the project (containers + empty volume) stopped, pushes each tar
    back into place with `docker cp` *before* anything starts, then starts and
    re-runs setup (which restarts monitor.sh — process state is not restored).
    """
    t0 = time.time()
    target_env.create()
    for service, _path, tar_name, dest_parent in CAPTURE:
        target_env.cp_in(service, dest_parent, ckpt_dir / tar_name)
    target_env.start()
    wait_for_gitea(target_env)
    target_env.token = token
    # Re-run the parts of setup that are process/not-file state. secret.txt,
    # repos and the Gitea DB all came back via the restored files above.
    start_monitor(target_env)
    return round(time.time() - t0, 3)
