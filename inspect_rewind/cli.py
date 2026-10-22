"""Command-line setup checks, checkpoint inspection, and branching."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import yaml


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def checkpoint_report(log_file: str, sample_id: str | None = None,
                      epoch: int | None = None) -> dict:
    from inspect_ai.log import read_eval_log
    from .fork import prefix_for_step, snapshots

    log = read_eval_log(log_file, resolve_attachments=True)
    samples = [s for s in log.samples or []
               if (sample_id is None or str(s.id) == sample_id)
               and (epoch is None or s.epoch == epoch)]
    if not samples:
        raise ValueError("no matching samples in this log")
    rows = []
    for sample in samples:
        calls = {tc.id: tc for m in sample.messages for tc in getattr(m, "tool_calls", None) or []}
        for step, entry in sorted(snapshots(sample).items()):
            call = calls.get(entry["tool_call_id"])
            try:
                prefix_for_step(sample.messages, entry["tool_call_id"])
                reason = None
            except ValueError as exc:
                reason = str(exc)
            available = (Path(entry["snapshot_dir"]) / "meta.json").is_file()
            rows.append({"sample": sample.id, "epoch": sample.epoch, "step": step,
                         "tool": call.function if call else "unknown",
                         "arguments": call.arguments if call else None,
                         "available": available, "forkable": reason is None,
                         "reason": reason, **entry})
    return {"log": log.location, "model": log.eval.model, "checkpoints": rows}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="inspect-rewind", description=(
        "Checkpoint and branch Inspect AI tasks with Docker sandboxes."), epilog=(
        "Run a task: inspect eval task.py --sandbox rewind --model PROVIDER/MODEL. "
        "Use 'inspect-rewind doctor' to check your setup."))
    ap.add_argument("--version", action="version", version=f"%(prog)s {version('rewind-env')}")
    ap.add_argument("--debug", action="store_true", help="show tracebacks for command errors")
    sub = ap.add_subparsers(dest="cmd", required=True)
    doctor = sub.add_parser("doctor", help="check local setup; no API key required")
    doctor.add_argument("--json", action="store_true", help="machine-readable output")
    listing = sub.add_parser("checkpoints", help="list samples and branch points in a log")
    listing.add_argument("log", help="parent .eval log")
    listing.add_argument("--sample", help="only this sample (default: all)")
    listing.add_argument("--epoch", type=positive_int, help="only this epoch (default: all)")
    listing.add_argument("--json", action="store_true", help="machine-readable output")
    exporting = sub.add_parser("export", help="bundle a checkpoint, transcript, and task for another machine")
    exporting.add_argument("log", help="parent .eval log")
    exporting.add_argument("--step", type=positive_int, required=True)
    exporting.add_argument("--sample", help="sample id (automatic for a single-sample log)")
    exporting.add_argument("--epoch", type=positive_int, default=1)
    exporting.add_argument("--output", type=Path, required=True, help="new .rewind bundle file")
    exporting.add_argument("--task-dir", type=Path, help=(
        "include local task source/data dependencies from this directory; "
        "default: the task file only (hidden files and run artifacts are excluded)"))
    importing = sub.add_parser("import", help="verify and unpack a portable Inspect bundle")
    importing.add_argument("bundle", type=Path)
    importing.add_argument("--output", type=Path, required=True, help="new directory for the restored run")
    f = sub.add_parser("fork", help="branch a sample at a saved step")
    f.add_argument("log", help="parent .eval log (run with --sandbox rewind)")
    f.add_argument("--sample", help="sample id (automatic for a single-sample log)")
    f.add_argument("--epoch", type=positive_int, default=1)
    f.add_argument("--step", type=positive_int, required=True, help="step from 'inspect-rewind checkpoints LOG'")
    f.add_argument("--model", help="model for the branch (default: the parent's)")
    g = f.add_mutually_exclusive_group()
    g.add_argument("--inject", help="user message appended after the parent's prefix")
    g.add_argument("--inject-file", type=Path, help="read the injected message from a file")
    f.add_argument("--mock-outputs", type=Path, help=argparse.SUPPRESS)
    f.add_argument("--log-dir", help="branch log directory (default: beside parent)")
    f.add_argument("--task", help="task file or registry name to use after moving a run")
    f.add_argument("--snapshot-dir", type=Path, help="relocated checkpoint directory containing meta.json")
    f.add_argument("--json", action="store_true", help="machine-readable branch report")
    return ap


def _execute(args: argparse.Namespace) -> int:
    if args.cmd == "doctor":
        from .doctor import doctor
        report = doctor()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            for check in report["checks"]:
                print(f"{'OK' if check['ok'] else 'FAIL'}  {check['name']}: {check['detail']}")
                if not check["ok"] and check.get("fix"):
                    print(f"      {check['fix']}")
        return 0 if report["ok"] else 1
    if args.cmd == "checkpoints":
        report = checkpoint_report(args.log, args.sample, args.epoch)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print(f"Model: {report['model']}")
            print(f"{'SAMPLE':<20} {'EPOCH':>5} {'STEP':>5} {'STATE':<14} TOOL / ARGUMENTS")
            for row in report["checkpoints"]:
                state = "missing files" if not row["available"] else "ready" if row["forkable"] else "partial turn"
                preview = json.dumps(row["arguments"], ensure_ascii=False)
                if len(preview) > 120:
                    preview = preview[:117] + "..."
                print(f"{str(row['sample']):<20} {row['epoch']:>5} {row['step']:>5} "
                      f"{state:<14} {row['tool']} {preview}")
            if not report["checkpoints"]:
                print("No checkpoints. Run the parent task with --sandbox rewind.")
        return 0

    if args.cmd == "export":
        from .bundle import export_bundle
        path = export_bundle(args.log, args.output, step=args.step, sample_id=args.sample,
                             epoch=args.epoch, task_dir=args.task_dir)
        print(f"BUNDLE={path}")
        return 0
    if args.cmd == "import":
        from .bundle import import_bundle
        path = import_bundle(args.bundle, args.output)
        print(f"PARENT_LOG={path}")
        return 0

    from .fork import fork, mock_outputs

    inject = args.inject_file.read_text() if args.inject_file else args.inject
    outputs = None
    if args.mock_outputs:
        spec = yaml.safe_load(args.mock_outputs.read_text())
        if not isinstance(spec, list) or not all(isinstance(item, dict) for item in spec):
            raise ValueError("--mock-outputs must contain a YAML list of output objects")
        outputs = mock_outputs(spec, args.model or "mockllm/model")
    log, report = fork(args.log, args.sample, args.step, epoch=args.epoch, model=args.model,
                       inject=inject, outputs=outputs, log_dir=args.log_dir,
                       task_spec=args.task, snapshot_dir=args.snapshot_dir,
                       display="none" if args.json else "plain")
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Branch {log.status}: checkpoint {args.step} -> {report['branch_model']}", file=sys.stderr)
        print(f"Restore verified: {bool(report.get('restore_ok'))}; "
              f"transcript preserved: {bool(report.get('prefix_ok'))}", file=sys.stderr)
        print(f"BRANCH_LOG={log.location}")
    ok = (log.status == "success" and report.get("restore_ok") and report.get("prefix_ok")
          and report.get("inject_ok", True))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _execute(args)
    except ImportError as exc:
        if args.debug:
            raise
        print(f"inspect-rewind: {exc}. Install the Inspect adapter with "
              "pip install 'rewind-env[inspect]' (or uv sync --extra inspect from a checkout).", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        if args.debug:
            raise
        print(f"inspect-rewind: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
