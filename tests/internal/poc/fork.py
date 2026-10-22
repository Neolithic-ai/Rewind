"""Rewind a run to step N and branch with an alternative script.

Starts a *fresh* environment (its own compose project = full isolation),
restores checkpoint N's files into it, seeds the transcript with the original
messages[:N], then replays the alternative steps.

Usage:  python -m tests.internal.poc.fork --run <run_id> --step N --script alt
Prints: BRANCH_ID=<id>
"""

from __future__ import annotations

import argparse
import json
import sys

from . import env as E
from .steps import Step, alt_steps
from .transcript import load_messages, run_script, save_messages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id produced by tests.internal.poc.task")
    ap.add_argument("--step", type=int, required=True, help="checkpoint/step N to rewind to")
    ap.add_argument("--script", default="alt", choices=["alt"], help="alternative script")
    args = ap.parse_args()

    run_dir = E.RUNS_DIR / args.run
    meta = json.loads((run_dir / "env.json").read_text())
    token = meta["token"]
    n = args.step

    branch_id = f"{args.run}_b{n}"
    branch_dir = E.RUNS_DIR / branch_id
    (branch_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    branch_env = E.Env(project=f"rewind_{branch_id}")
    print(f"[fork] branch_id={branch_id} project={branch_env.project}", file=sys.stderr)

    ckpt_dir = run_dir / "checkpoints" / f"step-{n}"
    print(f"[fork] restoring checkpoint {n} into fresh environment ...", file=sys.stderr)
    restore_seconds = E.restore(branch_env, ckpt_dir, token)
    print(f"[fork] restore took {restore_seconds}s", file=sys.stderr)

    # Seed the transcript with the original's messages up to step N
    # (1 user message + 2 messages per step), then replay the alt steps.
    prefix = load_messages(run_dir / "messages.json")[: 1 + 2 * n]
    steps = alt_steps(token)
    ckpt_metas: list[dict] = []

    def after_step(i: int, step: Step) -> None:
        m = E.snapshot(branch_env, n + i + 1, step.name,
                       branch_dir / "checkpoints" / f"step-{n + i + 1}")
        ckpt_metas.append(m)

    print(f"[fork] replaying {len(steps)} alternative steps ...", file=sys.stderr)
    messages = run_script(env=branch_env, steps=steps, log_dir=branch_dir / "log",
                          seed_messages=prefix, after_step=after_step)

    save_messages(messages, branch_dir / "messages.json")
    (branch_dir / "env.json").write_text(json.dumps({
        "branch_id": branch_id,
        "parent_run": args.run,
        "branched_from_step": n,
        "project": branch_env.project,
        "token": token,
        "restore_seconds": restore_seconds,
        "alt_steps": [s.name for s in steps],
        "checkpoints": ckpt_metas,
    }, indent=2))

    print(f"[fork] done. branch environment left running as {branch_env.project}",
          file=sys.stderr)
    print(f"BRANCH_ID={branch_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
