"""Run the original scripted agent and checkpoint after every step.

Brings up a fresh environment, provisions it (Gitea admin + token, secret.txt,
monitor.sh), replays the 6 scripted steps through Inspect's mock model, and saves
a checkpoint after each step with retention (the whole `runs/<id>/` tree is kept).

Usage:  python -m tests.internal.poc.task
Prints: RUN_ID=<id>   (consumed by run_all.sh)
"""

from __future__ import annotations

import datetime
import json
import sys

from . import env as E
from .steps import Step, original_steps
from .transcript import run_script, save_messages


def main() -> int:
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = E.RUNS_DIR / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    env = E.Env(project=f"rewind_{run_id}")
    print(f"[task] run_id={run_id} project={env.project}", file=sys.stderr)

    print("[task] bringing up environment ...", file=sys.stderr)
    env.up()
    E.wait_for_gitea(env)
    env.token = E.gitea_create_admin_and_token(env)
    E.install_monitor_and_secret(env)
    E.start_monitor(env)

    steps = original_steps(env.token)
    ckpt_metas: list[dict] = []

    def after_step(i: int, step: Step) -> None:
        n = i + 1
        meta = E.snapshot(env, n, step.name, run_dir / "checkpoints" / f"step-{n}")
        ckpt_metas.append(meta)
        print(f"[task] step {n} ({step.name}): checkpoint "
              f"{meta['disk_bytes']} B in {meta['save_seconds']}s", file=sys.stderr)

    print("[task] replaying 6 steps ...", file=sys.stderr)
    messages = run_script(env=env, steps=steps, log_dir=run_dir / "log",
                          after_step=after_step)

    save_messages(messages, run_dir / "messages.json")
    (run_dir / "env.json").write_text(json.dumps({
        "run_id": run_id,
        "project": env.project,
        "token": env.token,
        "steps": [s.name for s in steps],
        "checkpoints": ckpt_metas,
    }, indent=2))

    print(f"[task] done. original environment left running as {env.project}",
          file=sys.stderr)
    print(f"RUN_ID={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
