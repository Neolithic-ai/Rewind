"""Framework-independent commands for Docker filesystem checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any


def _doctor() -> dict[str, Any]:
    checks = []
    docker = shutil.which("docker")
    checks.append({"name": "docker_cli", "ok": docker is not None,
                   "detail": docker or "docker is not on PATH.",
                   "fix": "" if docker else "Install Docker Desktop or Docker Engine with Compose."})
    for name, args, fix in (
        ("docker_daemon", ["info", "--format", "{{.ServerVersion}}"],
         "Start Docker and check that this user can run docker info."),
        ("docker_compose", ["compose", "version", "--short"],
         "Install the Docker Compose v2 plugin."),
    ):
        ok, detail = False, "Not checked because docker is not on PATH."
        if docker:
            try:
                result = subprocess.run([docker, *args], capture_output=True, text=True, timeout=10)
                ok = result.returncode == 0
                detail = (result.stdout if ok else result.stderr).strip()[:500]
            except subprocess.TimeoutExpired:
                detail = "Command did not respond within 10 seconds."
            except OSError as exc:
                detail = str(exc)
        checks.append({"name": name, "ok": ok, "detail": detail, "fix": "" if ok else fix})
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def _capture(items: list[str]) -> dict[str, list[str]]:
    capture: dict[str, list[str]] = {}
    for item in items:
        service, separator, path = item.partition(":")
        if not separator or not service or not path.startswith("/"):
            raise ValueError(f"invalid capture {item!r}; use SERVICE:/absolute/container/path")
        capture.setdefault(service, []).append(path)
    return capture


def _state_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        raise ValueError(f"state file already exists: {path}; choose a new --state path")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_started(environment: Any, path: Path) -> None:
    try:
        environment.save(path)
    except BaseException:
        environment.down()
        raise


def _environment_report(environment: Any, state: Path) -> dict[str, Any]:
    report = {"ok": True, "state": str(state), "project": environment.project,
              "compose_file": str(environment.compose_file), "capture": environment.capture}
    if environment.restore_report is not None:
        report["restore"] = environment.restore_report
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rewind",
        description="Checkpoint and branch Docker filesystems with any agent framework.",
        epilog="Checkpoints preserve captured files. Processes restart; live sessions and IP addresses are not preserved.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('rewind-env')}")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor", help="check Docker prerequisites; no containers or API keys required")
    up = sub.add_parser("up", help="start an isolated Compose environment")
    up.add_argument("compose", type=Path, help="Docker Compose file")
    up.add_argument("--capture", action="append", required=True, metavar="SERVICE:/PATH",
                    help="container path to capture; repeat for more paths or services")
    up.add_argument("--state", type=Path, required=True, help="new file holding this environment's handle")
    checkpoint = sub.add_parser("checkpoint", help="capture a running environment's files")
    checkpoint.add_argument("state", type=Path)
    checkpoint.add_argument("--output", type=Path, required=True, help="new checkpoint directory")
    branch = sub.add_parser("branch", help="restore a checkpoint into a new isolated environment")
    branch.add_argument("checkpoint", type=Path)
    branch.add_argument("--state", type=Path, required=True, help="new file holding the branch's handle")
    branch.add_argument("--compose", type=Path, help="Compose file after moving the checkpoint to another machine")
    export = sub.add_parser("export", help="bundle a checkpoint and its environment configuration")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("--output", type=Path, required=True, help="new .rewind bundle file")
    unpack = sub.add_parser("import", help="verify and unpack a checkpoint bundle")
    unpack.add_argument("bundle", type=Path)
    unpack.add_argument("--output", type=Path, required=True, help="new directory for the imported checkpoint")
    execute = sub.add_parser("exec", help="run a command in an environment")
    execute.add_argument("state", type=Path)
    execute.add_argument("--service", default="default", help="Compose service (default: default)")
    execute.add_argument("command", nargs=argparse.REMAINDER, metavar="COMMAND",
                         help="command and arguments; optionally preceded by --")
    down = sub.add_parser("down", help="remove an environment's containers, networks and volumes")
    down.add_argument("state", type=Path)
    return parser


def _exec_command(args: argparse.Namespace) -> list[str]:
    # REMAINDER protects command flags from argparse, including flags named
    # --service. Accept our service selector immediately after STATE as well
    # as before it; everything after the command begins belongs to the command.
    command = list(args.command)
    if command and command[0] == "--service":
        if len(command) < 2:
            raise ValueError("--service requires a Compose service name")
        args.service = command[1]
        command = command[2:]
    elif command and command[0].startswith("--service="):
        args.service = command[0].partition("=")[2]
        command = command[1:]
    if command and command[0] == "--":
        command = command[1:]
    if not args.service or not command:
        raise ValueError("exec requires a command; use rewind exec STATE --service SERVICE -- COMMAND [ARGS...]")
    return command


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "doctor":
            report = _doctor()
            code = 0 if report["ok"] else 1
        else:
            # Keep help and prerequisite checks usable without loading any
            # evaluation framework or requiring a running Docker daemon.
            from .workspace import Checkpoint, Environment

            code = 0
            if args.action == "up":
                capture = _capture(args.capture)
                state = _state_path(args.state)
                environment = Environment.from_compose(args.compose, capture=capture)
                environment.start()
                _save_started(environment, state)
                report = _environment_report(environment, state)
            elif args.action == "checkpoint":
                environment = Environment.load(args.state)
                checkpoint = environment.checkpoint(args.output)
                report = {"ok": True, "checkpoint": str(checkpoint.path),
                          "project": environment.project, "manifest": checkpoint.manifest}
            elif args.action == "branch":
                state = _state_path(args.state)
                environment = Checkpoint.load(args.checkpoint).branch(compose_file=args.compose)
                _save_started(environment, state)
                report = _environment_report(environment, state)
            elif args.action == "exec":
                command = _exec_command(args)
                environment = Environment.load(args.state)
                rc, output = environment.exec(args.service, command)
                code = rc if 0 <= rc <= 255 else 1
                report = {"ok": rc == 0, "project": environment.project,
                          "service": args.service, "returncode": rc, "output": output}
            elif args.action == "export":
                bundle = Checkpoint.load(args.checkpoint).export(args.output)
                report = {"ok": True, "bundle": str(bundle)}
            elif args.action == "import":
                checkpoint = Checkpoint.import_bundle(args.bundle, args.output)
                report = {"ok": True, "checkpoint": str(checkpoint.path),
                          "compose_file": checkpoint.manifest["compose_file"]}
            else:
                environment = Environment.load(args.state)
                environment.down()
                environment.save(args.state)
                report = {"ok": True, "project": environment.project,
                          "state": str(args.state.expanduser().resolve()), "status": "removed"}
        print(json.dumps(report, indent=2, default=str))
        return code
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
