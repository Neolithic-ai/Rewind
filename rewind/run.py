"""Run a scenario's agent from scratch, checkpointing after every tool call.

Brings up a fresh environment, seeds it, takes checkpoint 0 (`seeded`), runs the
agent (mock path or real model) through Inspect with a checkpoint + milestone
detection after each tool call, scores the end state and records everything in
runs/<run_id>/ (messages.json, run.json, checkpoints/step-NNN/, log/).

Usage:
    python -m rewind.run --scenario ./scenario --path baseline
    python -m rewind.run --scenario ./scenario --model openrouter/provider/model
Prints: RUN_ID=<id>
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time

from .agent import MOCK_MODEL, detect_milestones, drive, fingerprint, initial_messages, is_mock, \
    mock_outputs, save_messages
from .env import RUNS_DIR, Env, exit_on_sigterm, seed, snapshot
from .openrouter import resolve_model
from .scenario import load_scenario


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", required=True, help="scenario directory")
    ap.add_argument("--path", default=None, help="mock path to replay (default: first)")
    ap.add_argument("--model", default=MOCK_MODEL,
                    help=f"Inspect model string (default {MOCK_MODEL}: replay a mock path)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="lower the scenario's agent.max_turns for this run")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="stop the agent once its model calls cost this much (if reported)")
    args = ap.parse_args()
    exit_on_sigterm()

    sc = load_scenario(args.scenario)
    mock = is_mock(args.model)
    path = None
    if mock:
        path = sc.mock_paths[args.path or next(iter(sc.mock_paths))]
        if path.start_from:
            ap.error(f"mock path {path.name!r} is a continuation from {path.start_from!r}; "
                     "use it with rewind.fork")
    else:
        resolve_model(args.model)  # fail fast (provider package / API key) before any container starts

    run_id = f"{sc.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = Env(sc, project=f"rw_{run_id}".replace("-", "_"))
    log = lambda s: print(f"[run] {s}", file=sys.stderr)  # noqa: E731
    log(f"run_id={run_id} project={env.project} "
        f"agent={'mock ' + path.name if mock else args.model}")

    t0 = time.time()
    try:
        env.up()
        env.wait_ready()
        variables = seed(env)
        log(f"environment up and seeded in {time.time() - t0:.1f}s")

        reached: set[str] = set()
        labels = detect_milestones(env, variables, reached)
        ckpt0 = snapshot(env, run_dir / "checkpoints" / "step-000", {
            "step": 0, "labels": labels, "reached": sorted(reached),
            "n_messages": len(initial_messages(sc)), "cmd": None, "rc": None,
            "fingerprint": fingerprint(env, variables)})
        log(f"  step  0 [{', '.join(labels)}]: save={ckpt0['save_seconds']}s "
            f"disk={ckpt0['disk_bytes'] / 1024:.0f}KiB")

        messages, ckpts, score, usage = drive(
            env=env, variables=variables, out_dir=run_dir, model=args.model,
            outputs=mock_outputs(path, model=args.model) if mock else None, seed_messages=None,
            start_step=0, reached=reached, log=log, max_turns=args.max_turns,
            budget_usd=args.budget_usd)

        save_messages(messages, run_dir / "messages.json")
        (run_dir / "run.json").write_text(json.dumps({
            "kind": "run",
            "run_id": run_id,
            "scenario": str(sc.root),
            "project": env.project,
            "model": args.model,
            "mock_path": path.name if path else None,
            "variables": variables,
            "checkpoints": [ckpt0, *ckpts],
            "score": score,
            "turns": sum(1 for m in messages if m.role == "assistant"),
            "usage": usage,
            "fingerprint": fingerprint(env, variables),
            "seconds": round(time.time() - t0, 1),
        }, indent=2))
    except BaseException:
        env.down()  # a crashed run leaves nothing behind (nobody else knows its project)
        raise
    log(f"done in {time.time() - t0:.1f}s; score success={score.get('success')}; "
        f"environment left running as {env.project}")
    print(f"RUN_ID={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
