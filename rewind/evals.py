"""Real-agent eval: N full attempts, then K verified branches from each fork point.

1. Runs `--attempts` full attempts of the scenario with `--model` (each one a
   `rewind.run`), scores them and records which milestones each one reached.
2. Picks a parent for each `--fork-from` point: the first successful attempt
   that reached it (else the first attempt that reached it). A fork point is a
   milestone label, a step number, or `mid` (the middle step of the parent's
   run, which every run of 2+ steps has). The default is `mid`, so a real run
   branches without requiring scenario-specific milestones. A point no
   attempt reached is reported as skipped, never dropped silently.
3. Forks `--branches-per-checkpoint` branches from each point (`rewind.fork`),
   round-robin over the points, and lets the model continue. Each branch is
   scored and verified:
   - restore content: every restored path matches the content hash recorded
     when the checkpoint was taken (checked before any service starts);
   - restore fingerprint: the restored environment's fingerprint equals the
     one recorded with the checkpoint (minus the scenario's volatile keys);
   - transcript: the branch's messages start with the parent's, up to the
     checkpoint;
   - isolation: once every branch has finished, each branch and each parent
     still has the fingerprint it ended with, so no branch touched another
     environment (and each parent's transcript file is unchanged);
   - model: the branch's Inspect log shows it called its own model, and only
     that one (for a variant with another model, not the parent's);
   - injected (variants with `inject`): the injected user message sits right
     after the parent's first N messages, and is marked as injected.
   With `--branch-variants FILE`, a YAML list of `{name, model?, inject?,
   inject_file?, system?}`, each fork point is branched once per variant (per
   `--branches-per-checkpoint`), and the summary has one row per fork point x
   variant. With a mock `--model`, a variant's real model is replaced by a
   differently named mock (`openrouter/x/y` -> `mockllm/y`), so the variant
   plumbing is tested without a key.
4. Writes `results.json` and `summary.md` to `runs/evals-<timestamp>/` (plus a
   copy of each run's run.json, transcript and Inspect log), and appends the
   summary to `$GITHUB_STEP_SUMMARY` if it is set. Tears every environment down
   at the end, also on errors and SIGTERM (`--keep` leaves them up).

Guards: every run is capped by the scenario's `agent.max_turns`; `--max-total-turns`
caps the turns of all runs together and `--budget-usd` their total cost (the
cost OpenRouter reports per call, or Inspect's own price data). A run is given
only what is left, and nothing new starts once either is used up.

With `--model mockllm/model` the attempts replay a mock path (no key), so the
whole pipeline can be tested without a model.

Usage:
    python -m rewind.evals --scenario ./scenario --model openrouter/provider/model --budget-usd 10
    python -m rewind.evals --scenario ./scenario --model mockllm/model --attempts 2
    python -m rewind.evals --scenario ./scenario --model openrouter/provider/model \
        --branch-variants ./variants.yaml --fork-from 3,mid
Prints: EVAL_DIR=<dir>. Exits non-zero if a run or fork crashed, any branch
check failed, or fewer than `--require-branches` (default 2) branches ran. An
agent that merely fails the task is a result, not an error.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml

from .agent import MARK, MOCK_MODEL, fingerprint, fingerprint_diff, is_mock, load_messages, \
    logged_models, normalize
from .env import RUNS_DIR, Env, exit_on_sigterm
from .fork import resolve_checkpoint
from .openrouter import require_key
from .scenario import load_scenario


def log(s: str) -> None:
    print(f"[evals] {s}", file=sys.stderr, flush=True)


VARIANT_KEYS = {"name", "model", "inject", "inject_file", "system"}


def load_variants(path: Path | None, base_model: str) -> list[dict]:
    """Branch variants from a YAML list of `{name, model?, inject?, inject_file?,
    system?}` (none: one unnamed variant, the plain branch). `model` defaults to
    `base_model`; with a mock `base_model`, a real model becomes the mock
    `mockllm/<its last path segment>`. `inject_file` is relative to the YAML."""
    if path is None:
        return [{"name": None, "model": base_model}]
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"{path}: expected a non-empty list of variants")
    out, names = [], set()
    for v in raw:
        if not isinstance(v, dict) or not v.get("name"):
            raise SystemExit(f"{path}: every variant needs a `name`: {v!r}")
        if unknown := set(v) - VARIANT_KEYS:
            raise SystemExit(f"{path}: variant {v['name']!r} has unknown keys {sorted(unknown)}")
        if v["name"] in names:
            raise SystemExit(f"{path}: duplicate variant name {v['name']!r}")
        if v.get("inject") is not None and v.get("inject_file"):
            raise SystemExit(f"{path}: variant {v['name']!r} has both inject and inject_file")
        names.add(v["name"])
        var = {"name": str(v["name"]), "model": v.get("model") or base_model,
               "inject": v.get("inject"), "system": v.get("system")}
        if v.get("inject_file"):
            var["inject_file"] = str((path.parent / v["inject_file"]).resolve())
            var["inject"] = Path(var["inject_file"]).read_text()
        if is_mock(base_model) and not is_mock(var["model"]):
            var["model_requested"] = var["model"]
            var["model"] = f"mockllm/{var['model'].rsplit('/', 1)[-1]}"
            log(f"variant {var['name']}: mock eval, so {var['model_requested']} "
                f"is stood in for by {var['model']}")
        out.append(var)
    return out


def variant_argv(v: dict) -> list[str]:
    """rewind.fork options for a variant (its model is passed separately)."""
    argv = ["--variant", v["name"]] if v["name"] else []
    if v.get("inject_file"):
        argv += ["--inject-file", v["inject_file"]]
    elif v.get("inject") is not None:
        argv += ["--inject", v["inject"]]
    if v.get("system") is not None:
        argv += ["--system", v["system"]]
    return argv


class Evals:
    def __init__(self, args: argparse.Namespace, out_dir: Path, variants: list[dict]) -> None:
        self.args = args
        self.out_dir = out_dir
        self.scenario = load_scenario(args.scenario)
        self.variants = variants
        self.turns_used = 0
        self.cost_used = 0.0
        self.cost_known = True
        self.stopped: str | None = None
        self.errors = 0
        self.envs: dict[str, Env] = {}  # run id -> environment still up

    # --- guards ------------------------------------------------------------
    def limits(self) -> list[str] | None:
        """CLI limits for the next run, or None (and self.stopped) if exhausted."""
        extra: list[str] = []
        if self.args.max_total_turns is not None:
            left = self.args.max_total_turns - self.turns_used
            if left <= 0:
                self.stopped = f"max total turns reached ({self.turns_used})"
                return None
            extra += ["--max-turns", str(left)]
        if self.args.budget_usd is not None:
            left_usd = self.args.budget_usd - self.cost_used
            if left_usd <= 0:
                self.stopped = f"budget reached (${self.cost_used:.4f})"
                return None
            extra += ["--budget-usd", f"{left_usd:.6f}"]
        return extra

    def account(self, rec: dict) -> None:
        self.turns_used += rec.get("turns") or 0
        cost = (rec.get("usage") or {}).get("cost_usd")
        if cost is not None:
            self.cost_used += cost
        elif rec.get("turns"):
            if self.cost_known and self.args.budget_usd is not None:
                log("WARNING: no cost reported for this model; --budget-usd cannot be "
                    "enforced, only the turn limits apply")
            self.cost_known = False

    # --- one run / branch -----------------------------------------------------
    def launch(self, module: str, argv: list[str], id_key: str, model: str | None = None) -> dict:
        limits = self.limits()
        if limits is None:
            log(f"SKIPPED: {self.stopped}")
            return {"skipped": self.stopped}
        cmd = [sys.executable, "-m", module, *argv, "--model", model or self.args.model, *limits]
        log(" ".join(cmd[1:]))
        t0 = time.time()
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, text=True,
            env={**os.environ, "REWIND_RUNS_DIR": str(RUNS_DIR)})
        wall = round(time.time() - t0, 1)
        run_id = next((line.split("=", 1)[1] for line in proc.stdout.splitlines()
                       if line.startswith(f"{id_key}=")), None)
        meta = None
        if run_id and (RUNS_DIR / run_id / "run.json").exists():
            meta = json.loads((RUNS_DIR / run_id / "run.json").read_text())
            self.envs[run_id] = Env(self.scenario, meta["project"])
            self.archive(RUNS_DIR / run_id)
        if meta and meta.get("aborted"):  # rewind.fork refused a restore that did not verify
            log(f"  {run_id}: {meta['aborted']}")
            return {"id": run_id, "aborted": meta["aborted"],
                    "restore_seconds": meta.get("restore_seconds"),
                    "restore_check": meta.get("restore_check"), "wall_seconds": wall}
        if proc.returncode != 0 or meta is None:
            self.errors += 1
            log(f"ERROR: {module} exited {proc.returncode}")
            return {"error": f"{module} exited {proc.returncode}", "wall_seconds": wall}
        rec = {
            "id": run_id,
            "success": bool(meta["score"].get("success")),
            "turns": meta.get("turns"),
            "usage": meta.get("usage"),
            "reached": reached(meta),
            "steps": steps_of(meta),
            "budget_exhausted": bool((meta.get("usage") or {}).get("budget_exhausted")),
            "wall_seconds": wall,
        }
        if meta["kind"] == "branch":
            rec |= {"restore_seconds": meta.get("restore_seconds"),
                    "restore_check": meta.get("restore_check")}
        self.account(rec)
        log(f"  {run_id}: success={rec['success']} turns={rec['turns']} "
            f"cost={fmt_cost(rec['usage'].get('cost_usd'))} wall={wall}s")
        return rec

    def archive(self, run_dir: Path) -> None:
        """Copy what a reviewer needs (not the checkpoint tars) into the eval dir."""
        dest = self.out_dir / "runs" / run_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("run.json", "messages.json"):
            if (run_dir / name).exists():
                shutil.copy2(run_dir / name, dest / name)
        if (run_dir / "log").is_dir():
            shutil.copytree(run_dir / "log", dest / "log", dirs_exist_ok=True)

    def down(self, run_id: str) -> None:
        env = self.envs.pop(run_id, None)
        if env and not self.args.keep:
            env.down()

    def teardown(self) -> None:
        if self.args.keep:
            log(f"leaving {len(self.envs)} environment(s) up (--keep)")
            return
        for run_id in list(self.envs):
            self.down(run_id)

    # --- fork points ----------------------------------------------------------
    def plan(self, attempts: list[dict]) -> list[dict]:
        """One entry per fork point: its parent and the --checkpoint to fork at,
        or why it is skipped."""
        done = [r for r in attempts if r.get("id")]
        points = []
        for cp in self.args.fork_from:
            if cp == "mid":
                cands = [r for r in done if len(r["steps"]) >= 2]
            elif cp.isdigit():
                cands = [r for r in done if int(cp) in r["steps"]]
            else:
                cands = [r for r in done if cp in r["reached"]]
            parent = next((r for r in cands if r["success"]), cands[0] if cands else None)
            if parent is None:
                why = ("no attempt ran" if not done else
                       "no attempt has 2+ steps" if cp == "mid" else "no attempt reached it")
                log(f"SKIPPED fork point {cp}: {why}")
                points.append({"point": cp, "parent": None, "skipped": why})
                continue
            at = str(max(parent["steps"]) // 2) if cp == "mid" else cp
            points.append({"point": cp, "parent": parent["id"], "checkpoint": at})
        return points

    # --- verification ---------------------------------------------------------
    def verify_branch(self, b: dict, parent_id: str, v: dict) -> dict:
        """Checks that need only the branch itself (isolation comes later)."""
        rc = b.get("restore_check") or {}
        checks = {
            "restore_content": (rc.get("content") or {}).get("ok"),
            "restore_fingerprint": (rc.get("fingerprint") or {}).get("ok"),
        }
        if b.get("aborted"):
            checks["transcript"] = None
            return checks
        meta = json.loads((RUNS_DIR / b["id"] / "run.json").read_text())
        n = meta["prefix_messages"]
        msgs = load_messages(RUNS_DIR / b["id"] / "messages.json")
        theirs = load_messages(RUNS_DIR / parent_id / "messages.json")[:n]
        override = meta.get("system_override")
        k = n  # where the parent's prefix ends in the branch
        system_ok = True
        if override is not None:  # the one message allowed to differ: compare the rest
            k += not any(m.role == "system" for m in theirs)  # an added system prompt
            sys_msgs = [m for m in msgs[:k] if m.role == "system"]
            system_ok = (len(sys_msgs) == 1 and sys_msgs[0].text == override
                         and (sys_msgs[0].metadata or {}).get(MARK) == "system_override")
            mine = [m for m in msgs[:k] if m.role != "system"]
            theirs = [m for m in theirs if m.role != "system"]
        else:
            mine = msgs[:n]
        checks["transcript"] = (len(msgs) >= k and system_ok
                                and normalize(mine) == normalize(theirs))
        if v.get("inject") is not None:
            m = msgs[k] if len(msgs) > k else None
            checks["injected"] = (meta.get("inject_index") == k and m is not None
                                  and m.role == "user" and m.text == v["inject"]
                                  and (m.metadata or {}).get(MARK) == "injected")
        # the Inspect log, not run.json, is the evidence of which model ran
        b["models_called"] = logged_models(RUNS_DIR / b["id"] / "log")
        called = b["models_called"]["inspect"]
        checks["model"] = called == [v["model"]] and (
            v["model"] == meta.get("parent_model") or meta.get("parent_model") not in called)
        return checks

    def transcript_hash(self, run_id: str) -> str:
        return hashlib.sha256((RUNS_DIR / run_id / "messages.json").read_bytes()).hexdigest()

    def isolation(self, run_id: str) -> tuple[bool | None, list[str] | None]:
        """Is this environment still in the state it ended in? (fingerprint diff)"""
        env = self.envs.get(run_id)
        meta = json.loads((RUNS_DIR / run_id / "run.json").read_text())
        if env is None or meta.get("aborted"):
            return None, None
        diff = fingerprint_diff(meta.get("fingerprint"), fingerprint(env, meta["variables"]),
                                self.scenario.fingerprint_volatile)
        return (None if diff is None else not diff), diff

    # --- the eval -----------------------------------------------------------
    def run(self) -> dict:
        a = self.args
        t0 = time.time()
        attempts = []
        for i in range(a.attempts):
            log(f"attempt {i + 1}/{a.attempts}")
            argv = ["--scenario", str(self.scenario.root)]
            if is_mock(a.model) and a.path:
                argv += ["--path", a.path]
            attempts.append(self.launch("rewind.run", argv, "RUN_ID"))

        points = self.plan(attempts)
        parents = {p["parent"] for p in points if p["parent"]}
        parent_hashes = {pid: self.transcript_hash(pid) for pid in parents}
        for r in attempts:  # attempts nobody branches from are done
            if r.get("id") and r["id"] not in parents:
                self.down(r["id"])
        if a.corrupt_checkpoint:
            target = next((p for p in points if p["parent"]), None)
            if target:
                corrupt(*resolve_checkpoint(RUNS_DIR / target["parent"], target["checkpoint"]))

        # round-robin, so a budget that runs out still leaves branches from
        # several fork points rather than all from the first one; each point
        # gets all its variants in turn, so every variant runs early on
        branches: list[dict] = []
        for k in range(a.branches_per_checkpoint):
            for p in points:
                if not p["parent"]:
                    continue
                for v in self.variants:
                    log(f"fork point {p['point']} (--checkpoint {p['checkpoint']}): branch "
                        f"{k + 1}/{a.branches_per_checkpoint} from {p['parent']}"
                        + (f", variant {v['name']} ({v['model']})" if v["name"] else ""))
                    argv = ["--run", p["parent"], "--checkpoint", p["checkpoint"],
                            *variant_argv(v)]
                    if is_mock(v["model"]) and a.fork_path:
                        argv += ["--path", a.fork_path]
                    b = self.launch("rewind.fork", argv, "BRANCH_ID", model=v["model"])
                    b |= {"point": p["point"], "parent": p["parent"], "variant": v["name"],
                          "model": v["model"]}
                    if b.get("id"):
                        meta = json.loads((RUNS_DIR / b["id"] / "run.json").read_text())
                        b["step"] = meta["branched_from_step"]
                        b["checks"] = self.verify_branch(b, p["parent"], v)
                    branches.append(b)

        # isolation: every environment still up must still look like it did
        # when it finished, after all the branches ran
        parent_isolation = {}
        for b in branches:
            if "checks" in b:
                b["checks"]["isolation"], b["isolation_diff"] = self.isolation(b["id"])
        for pid in sorted(parents):
            ok, diff = self.isolation(pid)
            parent_isolation[pid] = {"ok": ok, "diff": diff,
                                     "transcript_unchanged":
                                         self.transcript_hash(pid) == parent_hashes[pid]}

        ran = [b for b in branches if is_ran(b)]
        failed = [f"{b['id']}: {c}" for b in branches for c, ok in (b.get("checks") or {}).items()
                  if ok is not True and not (b.get("aborted") and ok is None)]
        failed += [f"parent {pid}: isolation" for pid, v in parent_isolation.items()
                   if v["ok"] is not True]
        failed += [f"parent {pid}: transcript changed" for pid, v in parent_isolation.items()
                   if not v["transcript_unchanged"]]
        reasons = []
        if len(ran) < a.require_branches:
            reasons.append(f"only {len(ran)} branch(es) ran, {a.require_branches} required")
        if failed:
            reasons.append(f"{len(failed)} check(s) failed")
        if self.errors:
            reasons.append(f"{self.errors} run(s) crashed")

        return {
            "scenario": self.scenario.name,
            "model": a.model,
            "started": datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            "verdict": {"ok": not reasons, "reasons": reasons, "failed_checks": failed,
                        "branches_ran": len(ran), "branches_required": a.require_branches},
            "attempts": attempts,
            "full_runs": stats(attempts),
            "reach": reach(attempts, self.scenario),
            "variants": self.variants,
            "fork_points": [p | stats([b for b in branches if b["point"] == p["point"]])
                            | {"variants": [{"variant": v["name"]}
                                            | stats([b for b in branches
                                                     if b["point"] == p["point"]
                                                     and b["variant"] == v["name"]])
                                            for v in self.variants]}
                            for p in points],
            "branches": branches,
            "parent_isolation": parent_isolation,
            "totals": {
                "turns": self.turns_used,
                "cost_usd": round(self.cost_used, 6) if self.cost_known else None,
                "wall_seconds": round(time.time() - t0, 1),
                "stopped": self.stopped,
                "errors": self.errors,
            },
        }


def corrupt(ckpt_dir: Path, ckpt: dict) -> None:
    """Testing: change one file inside the agent workstation's tar of a checkpoint
    (its recorded content hash stays), as a damaged checkpoint would be."""
    tar = next(p for p in sorted(ckpt_dir.glob("*.tar")) if p.name.startswith("default__"))
    tmp = tar.with_suffix(".corrupt")
    with tarfile.open(tar) as src, tarfile.open(tmp, "w") as dst:
        done = False
        for m in src.getmembers():
            data = src.extractfile(m).read() if m.isfile() else None
            if data is not None and not done:
                data += b"\n# corrupted by rewind.evals --corrupt-checkpoint\n"
                m.size = len(data)
                done = True
                log(f"CORRUPTING checkpoint {ckpt['step']}: {tar.name}:{m.name}")
            dst.addfile(m, None if data is None else io.BytesIO(data))
    tmp.replace(tar)


def is_ran(b: dict) -> bool:
    """A run/branch whose agent actually ran (not skipped, crashed or aborted)."""
    return "id" in b and not b.get("aborted")


def reached(meta: dict) -> list[str]:
    """Milestones reached in a run, including those of the checkpoint a branch
    started from."""
    labels = set(meta.get("branched_from_labels") or [])
    for c in meta["checkpoints"]:
        labels.update(c.get("reached", []))
    return sorted(labels)


def steps_of(meta: dict) -> list[int]:
    return [c["step"] for c in meta["checkpoints"]]


def reach(attempts: list[dict], scenario) -> dict:
    """Per milestone: how many full attempts reached it, and at which step."""
    ran = [r for r in attempts if is_ran(r)]
    out = {}
    for m in scenario.milestones:
        at = []
        for r in ran:
            meta = json.loads((RUNS_DIR / r["id"] / "run.json").read_text())
            at += [c["step"] for c in meta["checkpoints"] if m.name in c["labels"]]
        out[m.name] = {"reached": len(at), "of": len(ran),
                       "rate": round(len(at) / len(ran), 3) if ran else None, "steps": at}
    return out


def stats(recs: list[dict]) -> dict:
    """Success rate and resource totals over runs that actually ran."""
    ran = [r for r in recs if is_ran(r)]
    ok = sum(r["success"] for r in ran)
    costs = [r["usage"].get("cost_usd") for r in ran]
    return {
        "planned": len(recs),
        "ran": len(ran),
        "successes": ok,
        "success_rate": round(ok / len(ran), 3) if ran else None,
        "turns": sum(r["turns"] or 0 for r in ran),
        "total_tokens": sum(r["usage"].get("total_tokens", 0) for r in ran),
        "cost_usd": round(sum(costs), 6) if costs and None not in costs else None,
        "wall_seconds": round(sum(r["wall_seconds"] for r in ran), 1),
    }


def fmt_cost(c: float | None) -> str:
    return "n/a" if c is None else f"${c:.4f}"


def fmt_rate(s: dict) -> str:
    if s["success_rate"] is None:
        return "n/a"
    return f"{s['successes']}/{s['ran']} ({s['success_rate']:.0%})"


def fmt_check(ok: bool | None) -> str:
    return {True: "✅", False: "❌", None: "–"}[ok]


def fmt_variant(v: dict) -> str:
    return f"`{v['name']}`" if v["name"] else "–"


def markdown(res: dict, out_dir: Path) -> str:
    t, v = res["totals"], res["verdict"]
    verdict = (f"**PASS**: {v['branches_ran']} branches ran ({v['branches_required']} required), "
               "every restore, transcript, model, injection and isolation check passed."
               if v["ok"] else f"**FAIL**: {'; '.join(v['reasons'])}.")
    lines = [
        f"## Rewind evals: `{res['scenario']}` with `{res['model']}`",
        "",
        verdict,
        "",
        "| Runs | Variant | Parent | Success | Turns | Tokens | Cost | Wall time |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    rows = [("Full runs", "–", "", res["full_runs"])]
    for p in res["fork_points"]:
        name = (f"Branches from `{p['point']}`"
                + (f" (step {p['checkpoint']})" if p.get("checkpoint", p["point"]) != p["point"]
                   else ""))
        parent = f"`{p['parent']}`" if p["parent"] else f"**skipped: {p['skipped']}**"
        for var, vs in zip(res["variants"], p["variants"]):
            rows.append((name, fmt_variant(var), parent, vs))
    for name, variant, parent, s in rows:
        ran = f" ({s['ran']}/{s['planned']} ran)" if s["ran"] != s["planned"] else ""
        lines.append(f"| {name}{ran} | {variant} | {parent} | {fmt_rate(s)} | {s['turns']} | "
                     f"{s['total_tokens']:,} | {fmt_cost(s['cost_usd'])} | {s['wall_seconds']}s |")
    if any(var["name"] for var in res["variants"]):
        lines += ["", "Variants: " + "; ".join(
            f"`{var['name']}` = `{var['model']}`"
            + (f" (standing in for `{var['model_requested']}`)" if var.get("model_requested")
               else "")
            + (f", inject {json.dumps(var['inject'])}" if var.get("inject") is not None else "")
            + (f", system {json.dumps(var['system'])}" if var.get("system") is not None else "")
            for var in res["variants"]) + "."]

    lines += ["", "### Milestones reached by the full runs", "",
              "| Milestone | Reached | Rate | At step |", "| --- | ---: | ---: | --- |"]
    for m, r in res["reach"].items():
        rate = "n/a" if r["rate"] is None else f"{r['rate']:.0%}"
        lines.append(f"| `{m}` | {r['reached']}/{r['of']} | {rate} | "
                     f"{', '.join(map(str, r['steps'])) or '–'} |")

    lines += ["", "### Branches", "",
              "| Branch | Variant | From | Restore | Content hash | Fingerprint | Transcript "
              "| Injected | Model | Isolation | Success | Turns | Cost | Wall time |",
              "| --- | --- | --- | ---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | ---: | ---: "
              "| ---: |"]
    for b in res["branches"]:
        frm = f"`{b['point']}`" + (f" (step {b['step']})" if "step" in b else "")
        variant = f"`{b['variant']}`" if b.get("variant") else "–"
        if "id" not in b:
            why = b.get("skipped") or b.get("error")
            lines.append(f"| **not run: {why}** | {variant} | {frm} |" + " |" * 11)
            continue
        c = b["checks"]
        ok = is_ran(b)
        called = ", ".join(f"`{m}`" for m in (b.get("models_called") or {}).get("inspect", []))
        lines.append(
            f"| `{b['id']}`{' **(aborted)**' if b.get('aborted') else ''} | {variant} | {frm} | "
            f"{b.get('restore_seconds') or '–'}s | {fmt_check(c['restore_content'])} | "
            f"{fmt_check(c['restore_fingerprint'])} | {fmt_check(c['transcript'])} | "
            f"{fmt_check(c.get('injected'))} | "
            f"{fmt_check(c.get('model'))}{' ' + called if called else ''} | "
            f"{fmt_check(c['isolation'])} | {fmt_check(b['success']) if ok else '–'} | "
            f"{b['turns'] if ok else '–'} | {fmt_cost(b['usage'].get('cost_usd')) if ok else '–'} | "
            f"{b['wall_seconds']}s |")
    if not res["branches"]:
        lines.append("| **no branches ran** |" + " |" * 13)
    for pid, iso in res["parent_isolation"].items():
        lines.append(f"\nParent `{pid}` unchanged after its branches: {fmt_check(iso['ok'])}"
                     + (f" (differs in {iso['diff']})" if iso["diff"] else "")
                     + f"; transcript unchanged: {fmt_check(iso['transcript_unchanged'])}")

    lines += [
        "",
        f"Total: {t['turns']} turns, {fmt_cost(t['cost_usd'])}, {t['wall_seconds']}s wall time.",
    ]
    if t["stopped"]:
        lines.append(f"Stopped early: {t['stopped']}.")
    if t["errors"]:
        lines.append(f"**{t['errors']} run(s) crashed**; see the job log.")
    if v["failed_checks"]:
        lines.append("Failed checks: " + ", ".join(f"`{f}`" for f in v["failed_checks"]))
    lines += ["", f"Results: `{out_dir / 'results.json'}`", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True,
                    help=f"Inspect model string, e.g. openrouter/anthropic/claude-sonnet-5 "
                         f"or {MOCK_MODEL} (replays mock paths)")
    ap.add_argument("--scenario", required=True, help="scenario directory")
    ap.add_argument("--attempts", type=int, default=3, help="full runs (default 3)")
    ap.add_argument("--fork-from", default="mid",
                    help="comma-separated fork points: milestone labels, step numbers or "
                         "`mid` (default mid)")
    ap.add_argument("--branches-per-checkpoint", type=int, default=2,
                    help="branches per fork point (default 2)")
    ap.add_argument("--require-branches", type=int, default=2,
                    help="fail unless at least this many branches ran (default 2)")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="stop once the runs together have cost this much")
    ap.add_argument("--max-total-turns", type=int, default=None,
                    help="stop once the runs together have used this many turns")
    ap.add_argument("--branch-variants", type=Path, default=None,
                    help="YAML list of {name, model?, inject?, inject_file?, system?}: branch "
                         "each fork point once per variant")
    ap.add_argument("--path", default=None, help="mock model only: path for the full runs")
    ap.add_argument("--fork-path", default=None,
                    help="mock model only: path for the branches (default: the parent's)")
    ap.add_argument("--corrupt-checkpoint", action="store_true",
                    help="testing: damage the first fork point's checkpoint before branching "
                         "(the eval must then fail its restore check)")
    ap.add_argument("--keep", action="store_true",
                    help="leave the Docker environments running")
    args = ap.parse_args()
    args.fork_from = [c.strip() for c in args.fork_from.split(",") if c.strip()]
    require_key(args.model)
    variants = load_variants(args.branch_variants, args.model)
    for v in variants:  # fail fast, before any container starts
        require_key(v["model"])
    exit_on_sigterm()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RUNS_DIR / f"evals-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"eval dir {out_dir}")

    ev = Evals(args, out_dir, variants)
    try:
        res = ev.run()
    finally:
        ev.teardown()
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    md = markdown(res, out_dir)
    (out_dir / "summary.md").write_text(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md + "\n")
    print(md)
    print(f"EVAL_DIR={out_dir}")
    return 0 if res["verdict"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
