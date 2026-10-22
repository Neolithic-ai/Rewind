"""Portable Inspect logs, captured files, and explicitly selected task sources."""

from __future__ import annotations

import ast
import copy
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml
from inspect_ai.event import InfoEvent
from inspect_ai.log import EvalLog, read_eval_log, write_eval_log
from inspect_ai.util._sandbox.docker.docker import resolve_config_environment

from rewind.bundle import extracted_bundle, publish_directory, stage_compose, write_bundle
from rewind.env import tar_digest, tar_name

from .fork import find_sample, prefix_for_step, snapshots
from .sandbox import SOURCE


def _resolve_file(value: str, log: EvalLog, relative_to: Path | None = None) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [*(
        [relative_to / path] if relative_to is not None else []), Path.cwd() / path, *(
        parent / path for parent in Path(log.location).resolve().parents)]
    found = next((p.resolve() for p in candidates if p.is_file()), None)
    if found is None:
        raise FileNotFoundError(f"bundle source file is unavailable: {value}")
    return found


def _relative_file(root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("bundle file reference must be a relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts or "\\" in value:
        raise ValueError(f"invalid bundle file reference: {value!r}")
    resolved = root / path
    if not resolved.is_file():
        raise ValueError(f"bundle is missing required file: {value}")
    return resolved


_EXCLUDED = {"__pycache__", "node_modules", "venv", "runs", "logs", "snapshots", "checkpoints", "dist", "build"}
_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def _task_sources(task_file: Path, destination: Path, task_dir: Path | None) -> Path:
    """Copy a single task, or an explicitly selected source directory; never import it."""
    destination.mkdir(parents=True)
    if task_dir is None:
        tree = ast.parse(task_file.read_text(), filename=str(task_file))
        for node in ast.walk(tree):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("task has relative imports; include its source with --task-dir PATH")
            for name in names:
                first = name.split(".")[0]
                if (task_file.parent / f"{first}.py").is_file() or (task_file.parent / first).is_dir():
                    raise ValueError(f"task imports local module {name!r}; include its source "
                                     "with --task-dir PATH")
        target = destination / task_file.name
        shutil.copy2(task_file, target)
        return target
    source = task_dir.expanduser().resolve()
    if not source.is_dir() or not task_file.is_relative_to(source):
        raise ValueError("--task-dir must be a directory containing the task file")
    for directory, directories, files in os.walk(source):
        directories[:] = [name for name in directories if not name.startswith(".") and name not in _EXCLUDED]
        for name in [*directories, *files]:
            path = Path(directory) / name
            relative = path.relative_to(source)
            if name.startswith(".") or path.suffix in _SECRET_SUFFIXES | {".eval", ".rewind", ".pyc", ".env"}:
                continue
            if path.is_symlink():
                raise ValueError(f"task sources contain a symlink: {relative}; provide regular source files")
            if path.is_file():
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    target = destination / task_file.relative_to(source)
    if not target.is_file():
        raise ValueError("the task file is excluded from portable sources; select a normal source directory")
    return target


def _verify_checkpoint(directory: Path, entry: dict) -> dict:
    try:
        meta = json.loads((directory / "meta.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read checkpoint metadata at {directory}: {exc}") from exc
    expected = entry.get("content_sha256")
    if not isinstance(expected, dict) or not expected or not isinstance(meta, dict) or (
        meta.get("content_sha256") != expected
    ):
        raise ValueError(f"checkpoint {directory} does not match its transcript hashes")
    for name, digest in expected.items():
        if not isinstance(name, str) or Path(name).name != name or "\\" in name or not name.endswith(".tar"):
            raise ValueError(f"invalid checkpoint archive name: {name!r}")
        archive = directory / name
        try:
            verified = archive.is_file() and not archive.is_symlink() and tar_digest(archive) == digest
        except (OSError, tarfile.TarError) as exc:
            raise ValueError(f"cannot read checkpoint archive: {archive}") from exc
        if not verified:
            raise ValueError(f"checkpoint archive failed hash verification: {archive}")
    return meta


def _write_task_compose(config: dict, canonical: Path, task_compose: Path) -> None:
    """Keep task-side relative config lookups valid without duplicating assets."""
    task_config = copy.deepcopy(config)

    def relative_asset(value: str) -> str:
        return os.path.relpath(canonical.parent / value, task_compose.parent)

    for spec in task_config.get("services", {}).values():
        for mount in spec.get("volumes") or []:
            if mount.get("type") == "bind":
                mount["source"] = relative_asset(mount["source"])
    for kind in ("configs", "secrets"):
        for spec in (task_config.get(kind) or {}).values():
            if spec.get("file"):
                spec["file"] = relative_asset(spec["file"])
    task_compose.parent.mkdir(parents=True, exist_ok=True)
    task_compose.write_text(yaml.safe_dump(task_config, sort_keys=False))


def _relocate_events(log: EvalLog, entries: dict[int, dict], root: Path | None = None) -> None:
    """Keep only bundled checkpoints, assigning their portable or installed paths."""
    def relocated(entry: dict) -> dict:
        result = dict(entry)
        if root is not None:
            result["snapshot_dir"] = str(root / result["snapshot_dir"])
        return result

    for sample in log.samples or []:
        events = []
        for event in sample.events:
            if not (isinstance(event, InfoEvent) and event.source == SOURCE
                    and isinstance(event.data, dict)):
                events.append(event)
                continue
            kind = event.data.get("rewind")
            if kind == "snapshot":
                step = int(event.data["step"])
                if step not in entries:
                    continue
                event.data = {"rewind": "snapshot", **relocated(entries[step])}
            elif kind == "branch":
                event.data = {**event.data, "inherited": {
                    str(step): relocated(entries[step]) for step in entries
                    if str(step) in (event.data.get("inherited") or {})}}
            events.append(event)
        sample.events = events


def export_bundle(log_file: str | Path, output: str | Path, *, step: int,
                  sample_id: str | int | None = None, epoch: int = 1,
                  task_dir: str | Path | None = None) -> Path:
    """Package one selected sample and its available checkpoint ancestry."""
    parent = read_eval_log(str(log_file), resolve_attachments=True)
    sample = find_sample(parent, sample_id, epoch)
    available = snapshots(sample)
    if step not in available:
        raise ValueError(f"no snapshot for step {step}; available steps: {sorted(available)}")
    prefix_for_step(sample.messages, available[step]["tool_call_id"])
    if not parent.eval.task_file:
        raise ValueError("portable export requires a file-based task; registry-only tasks cannot be bundled")
    task_file = _resolve_file(parent.eval.task_file, parent)
    sandbox = sample.sandbox if sample.sandbox and sample.sandbox.config is not None else parent.eval.sandbox
    if sandbox is None or not isinstance(sandbox.config, str):
        raise ValueError("portable export requires an explicit Docker Compose YAML configuration")
    compose_file = _resolve_file(sandbox.config, parent, task_file.parent)
    if compose_file.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("portable export requires Docker Compose YAML; wrap Dockerfile configs in Compose")

    with tempfile.TemporaryDirectory(prefix="rewind-inspect-export-") as temporary:
        staging = Path(temporary)
        task_path = _task_sources(task_file, staging / "task", Path(task_dir) if task_dir else None)
        resolved = resolve_config_environment(str(compose_file), sample.metadata or {})
        compose = stage_compose(compose_file, staging / "environment",
                                environment=resolved.env if resolved else None)
        entries: dict[int, dict] = {}
        missing = []
        for number, entry in sorted(available.items()):
            if number > step:
                continue
            source = Path(entry["snapshot_dir"])
            if number != step and not (source / "meta.json").is_file():
                missing.append(number)
                continue
            _verify_checkpoint(source, entry)
            target = staging / "checkpoints" / f"step-{number:03d}"
            target.mkdir(parents=True)
            for name in ["meta.json", *entry["content_sha256"]]:
                shutil.copy2(source / name, target / name)
            entries[number] = {**entry, "snapshot_dir": target.relative_to(staging).as_posix()}

        # Preserve explicit capture settings, including runs configured through env vars.
        capture = next((event.data["capture"] for event in sample.events
                        if isinstance(event, InfoEvent) and event.source == SOURCE
                        and isinstance(event.data, dict) and event.data.get("rewind") == "capture"), None)
        if not isinstance(capture, dict) or {
            tar_name(service, path) for service, paths in capture.items() for path in paths
        } != set(entries[step]["content_sha256"]):
            raise ValueError("checkpoint capture configuration is missing or differs from the selected snapshot")
        config = yaml.safe_load(compose.read_text())
        for service, spec in config["services"].items():
            spec["x-rewind"] = {"capture": capture.get(service, [])}
        compose.write_text(yaml.safe_dump(config, sort_keys=False))
        # Task constructors often retain a relative sandbox filename. Make that file
        # available beside their source as well; the log points at the portable config.
        original_relative = compose_file.relative_to(task_file.parent).as_posix() if (
            compose_file.is_relative_to(task_file.parent)
        ) else (parent.eval.metadata or {}).get("rewind_bundle", {}).get("task_compose")
        if original_relative:
            relative = PurePosixPath(original_relative)
            if relative.is_absolute() or PureWindowsPath(original_relative).drive or ".." in relative.parts:
                raise ValueError("invalid task Compose reference in bundle metadata")
            _write_task_compose(config, compose, task_path.parent / original_relative)
            parent.eval.metadata = {**(parent.eval.metadata or {}),
                                    "rewind_bundle": {"task_compose": original_relative}}
        parent.samples = [sample]
        parent.eval.task_file = task_path.relative_to(staging).as_posix()
        parent.eval.sandbox = sandbox.model_copy(update={"config": compose.relative_to(staging).as_posix()})
        if sample.sandbox is not None:
            sample.sandbox = parent.eval.sandbox.model_copy()
        _relocate_events(parent, entries)
        write_eval_log(parent, staging / "parent.eval")
        return write_bundle(staging, Path(output), {
            "kind": "inspect", "log": "parent.eval", "task_file": parent.eval.task_file,
            "compose_file": parent.eval.sandbox.config, "step": step,
            "sample_id": sample.id, "epoch": epoch, "checkpoints": sorted(entries),
            "unavailable_ancestor_steps": missing,
            "task_source_mode": "directory" if task_dir else "single_file",
        })


def import_bundle(archive: str | Path, output: str | Path) -> Path:
    """Verify and relocate an Inspect bundle without loading or running task code."""
    destination = Path(output).expanduser().resolve()
    with extracted_bundle(Path(archive)) as (staging, manifest):
        if manifest.get("kind") != "inspect":
            raise ValueError("expected an Inspect bundle; use rewind import for standalone checkpoints")
        log_file = _relative_file(staging, manifest.get("log"))
        task_file = _relative_file(staging, manifest.get("task_file"))
        compose_file = _relative_file(staging, manifest.get("compose_file"))
        log = read_eval_log(str(log_file), resolve_attachments=True)
        sample = find_sample(log, manifest.get("sample_id"), manifest.get("epoch", 1))
        entries = snapshots(sample)
        step = manifest.get("step")
        if step not in entries or sorted(entries) != manifest.get("checkpoints"):
            raise ValueError("bundle checkpoint index does not match the saved transcript")
        prefix_for_step(sample.messages, entries[step]["tool_call_id"])
        for entry in entries.values():
            meta = _relative_file(staging, f"{entry['snapshot_dir']}/meta.json")
            _verify_checkpoint(meta.parent, entry)
        if log.eval.sandbox is None:
            raise ValueError("bundle log is missing sandbox configuration")
        log.eval.task_file = str(destination / task_file.relative_to(staging))
        log.eval.sandbox = log.eval.sandbox.model_copy(update={
            "config": str(destination / compose_file.relative_to(staging))})
        if sample.sandbox is not None:
            sample.sandbox = log.eval.sandbox.model_copy()
        _relocate_events(log, entries, destination)
        write_eval_log(log, log_file)
        publish_directory(staging, destination)
    return destination / manifest["log"]
