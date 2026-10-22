"""Rewind a run to a checkpoint (milestone label or step number) and branch.

Restores the checkpoint into a *fresh* compose project (full isolation from the
original) and verifies it: the restored files must match the content hashes
recorded at the checkpoint, and the restored environment's fingerprint must
match the one recorded there (minus the scenario's `volatile` keys). On a
mismatch it records `restore_check` in run.json, does not run the agent and
exits 3. Otherwise it seeds the transcript with the parent's messages up to that
checkpoint, then continues with a mock path or a real model, checkpointing and
scoring exactly like `rewind.run`.

Branch variants change what the branch continues with:

- `--model M`: a different model from the parent's (the default is the
  parent's). If any of the parent's messages came from another model (here or
  in an ancestor), they are adapted for the new provider first
  (`agent.adapt_for_model`: reasoning blocks dropped, tool-call ids renumbered,
  provider-specific content removed). run.json records `parent_model`,
  `branch_model` and what the adaptation changed (`prefix_adapted`).
- `--inject TEXT` / `--inject-file PATH`: a new user message appended after the
  parent's first N messages, before the model continues.
- `--system TEXT`: replaces the branch's system prompt.

Injected and overriding messages are marked in their metadata (`rewind:
injected` / `rewind: system_override`), recorded in run.json (`inject`,
`inject_index`, `system_override`) and noted in the Inspect log as an info event.

Usage:
    python -m rewind.fork --run <id> --checkpoint 7 --model <provider/model>
    python -m rewind.fork --run <id> --checkpoint <label> --model <provider/model> \
        --inject "Continue with the revised instructions."
Prints: BRANCH_ID=<id>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
import time
from pathlib import Path

from .agent import adapt_for_model, branch_prefix, drive, fingerprint, fingerprint_diff, is_mock, \
    load_messages, mock_outputs, save_messages
from .env import RUNS_DIR, Env, exit_on_sigterm, restore
from .openrouter import resolve_model
from .scenario import load_scenario


def resolve_checkpoint(run_dir: Path, which: str) -> tuple[Path, dict]:
    """Find a checkpoint in this branch's history, excluding ancestors' futures."""
    current = run_dir
    max_step: int | None = None
    labels = []
    while True:
        meta = json.loads((current / "run.json").read_text())
        for c in meta["checkpoints"]:
            if max_step is not None and c["step"] > max_step:
                continue
            labels.append(f"{c['step']}:{','.join(c['labels'])}")
            if which in c["labels"] or (which.isdigit() and c["step"] == int(which)):
                return current / "checkpoints" / f"step-{c['step']:03d}", c
        if not meta.get("parent_run"):
            break
        fork_step = meta["branched_from_step"]
        max_step = fork_step if max_step is None else min(max_step, fork_step)
        current = RUNS_DIR / meta["parent_run"]
    raise SystemExit(f"no checkpoint {which!r} in {run_dir.name}; have {labels}")


def lineage_models(meta: dict) -> set[str]:
    """Every model that wrote part of a run's transcript (a branch inherits its
    ancestors' messages)."""
    models = {meta.get("model")}
    if meta.get("parent_run"):
        models |= lineage_models(json.loads((RUNS_DIR / meta["parent_run"] / "run.json").read_text()))
    return models


def project_name(branch_id: str, limit: int = 60) -> str:
    """Compose project for a branch; long ids keep a hash so they stay unique."""
    name = f"rw_{branch_id}".replace("-", "_")
    if len(name) <= limit:
        return name
    return f"{name[:limit - 9]}_{hashlib.sha1(branch_id.encode()).hexdigest()[:8]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="run (or branch) id to fork from")
    ap.add_argument("--checkpoint", required=True, help="milestone label or step number")
    ap.add_argument("--path", default=None,
                    help="mock path to continue with (default: the parent's own path)")
    ap.add_argument("--model", default=None,
                    help="Inspect model string (default: the parent's model)")
    inj = ap.add_mutually_exclusive_group()
    inj.add_argument("--inject", default=None,
                     help="append this user message after the checkpoint's messages")
    inj.add_argument("--inject-file", type=Path, default=None,
                     help="like --inject, with the message read from a file")
    ap.add_argument("--system", default=None, help="replace the branch's system prompt")
    ap.add_argument("--variant", default=None,
                    help="name of this branch variant (recorded, and part of the branch id)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="lower the scenario's agent.max_turns for this branch")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="stop the agent once its model calls cost this much (if reported)")
    args = ap.parse_args()
    exit_on_sigterm()

    run_dir = RUNS_DIR / args.run
    parent = json.loads((run_dir / "run.json").read_text())
    sc = load_scenario(parent["scenario"])
    ckpt_dir, ckpt = resolve_checkpoint(run_dir, args.checkpoint)
    n = ckpt["step"]
    parent_model = parent.get("model")
    model = args.model or parent_model
    inject = args.inject_file.read_text() if args.inject_file else args.inject

    mock = is_mock(model)
    outputs = None
    path = None
    if mock:
        path = sc.mock_paths[args.path or parent.get("mock_path") or next(iter(sc.mock_paths))]
        if path.start_from:
            if path.start_from not in ckpt["reached"]:
                raise SystemExit(f"path {path.name} continues from {path.start_from!r}, "
                                 f"but checkpoint {n} has only reached {ckpt['reached']}")
            outputs = mock_outputs(path, model=model)
        else:
            outputs = mock_outputs(path, skip=n, model=model)  # replay the rest of a full script
    else:
        resolve_model(model)  # fail fast (provider package / API key) before any container starts

    agent_name = path.name if mock else re.sub(r"[^a-z0-9]+", "-", model.lower())
    label = ckpt["labels"][-1] if ckpt["labels"] else f"step{n}"
    branch_id = base_id = f"{args.run}__{label}-{agent_name}" + (
        f"-{re.sub(r'[^a-z0-9]+', '-', args.variant.lower())}" if args.variant else "")
    for i in itertools.count(2):  # several branches from one checkpoint: -2, -3, ...
        branch_dir = RUNS_DIR / branch_id
        try:
            branch_dir.mkdir(parents=True)
            break
        except FileExistsError:
            branch_id = f"{base_id}-{i}"
    env = Env(sc, project=project_name(branch_id))
    log = lambda s: print(f"[fork] {s}", file=sys.stderr)  # noqa: E731
    log(f"branch_id={branch_id} project={env.project} from step {n} "
        f"{ckpt['labels'] or ''} agent={'mock ' + path.name if mock else model}"
        + (f" model={model} (parent {parent_model})" if model != parent_model else "")
        + (" +inject" if inject is not None else "") + (" +system" if args.system else ""))

    t0 = time.time()
    record = {
        "kind": "branch",
        "run_id": branch_id,
        "parent_run": args.run,
        "branched_from_step": n,
        "branched_from_labels": ckpt["labels"],
        "prefix_messages": ckpt["n_messages"],
        "scenario": str(sc.root),
        "project": env.project,
        "model": model,
        "parent_model": parent_model,
        "branch_model": model,
        "variant": args.variant,
        "inject": inject,
        "inject_file": str(args.inject_file) if args.inject_file else None,
        "system_override": args.system,
        "mock_path": path.name if path else None,
        "variables": parent["variables"],
    }
    try:
        restore_seconds, content = restore(env, ckpt_dir)
        variables = parent["variables"]
        at_restore = fingerprint(env, variables)
        diff = fingerprint_diff(ckpt.get("fingerprint"), at_restore, sc.fingerprint_volatile)
        check = {
            "content": content,
            "fingerprint": {"ok": None if diff is None else not diff, "diff": diff,
                            "ignored": sc.fingerprint_volatile},
        }
        check["ok"] = content["ok"] is not False and check["fingerprint"]["ok"] is not False
        record |= {"restore_seconds": restore_seconds, "restore_check": check,
                   "fingerprint_at_restore": at_restore}
        log(f"restored into fresh environment in {restore_seconds}s; content hash "
            f"{fmt_ok(content['ok'])}, fingerprint {fmt_ok(check['fingerprint']['ok'])}")
        if not check["ok"]:
            bad = [k for k, v in content["paths"].items() if v["ok"] is False]
            log(f"ERROR: restored environment does not match checkpoint {n}: "
                f"content differs in {bad or 'nothing'}, fingerprint differs in {diff or 'nothing'}; "
                "not running the agent")
            (branch_dir / "run.json").write_text(json.dumps(
                record | {"aborted": "restore verification failed",
                          "seconds": round(time.time() - t0, 1)}, indent=2))
            print(f"BRANCH_ID={branch_id}")
            return 3

        prefix = load_messages(run_dir / "messages.json")[: ckpt["n_messages"]]
        if lineage_models(parent) - {model}:  # some of the prefix came from another model
            prefix, adapted = adapt_for_model(prefix)
            record["prefix_adapted"] = adapted
            log(f"adapted the parent's messages for {model}: {adapted}")
        seed, inject_index = branch_prefix(prefix, system=args.system, inject=inject)
        record["inject_index"] = inject_index
        note = {k: record[k] for k in ("parent_run", "branched_from_step", "parent_model",
                                       "branch_model", "variant", "inject", "inject_index",
                                       "system_override")}
        messages, ckpts, score, usage = drive(
            env=env, variables=variables, out_dir=branch_dir, model=model,
            outputs=outputs, seed_messages=seed, start_step=n,
            reached=set(ckpt["reached"]), log=log, max_turns=args.max_turns,
            budget_usd=args.budget_usd, note={"rewind_branch": note})

        save_messages(messages, branch_dir / "messages.json")
        (branch_dir / "run.json").write_text(json.dumps(record | {
            "checkpoints": ckpts,
            "score": score,
            "turns": sum(1 for m in messages[ckpt["n_messages"]:] if m.role == "assistant"),
            "usage": usage,
            "fingerprint": fingerprint(env, variables),
            "seconds": round(time.time() - t0, 1),
        }, indent=2))
    except BaseException:
        env.down()  # a crashed branch leaves nothing behind (nobody else knows its project)
        raise
    log(f"done in {time.time() - t0:.1f}s; score success={score.get('success')}; "
        f"environment left running as {env.project}")
    print(f"BRANCH_ID={branch_id}")
    return 0


def fmt_ok(ok: bool | None) -> str:
    return {True: "verified", False: "MISMATCH", None: "not recorded"}[ok]

if __name__ == "__main__":
    raise SystemExit(main())
