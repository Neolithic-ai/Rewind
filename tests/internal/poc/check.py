"""Verify a branch against its parent run and print a PASS/FAIL table.

Checks the branched environment, the untouched original environment, and the two
transcripts, then prints timings and per-checkpoint disk usage. Exits non-zero if
any check's actual result differs from its expected result.

Usage:  python -m tests.internal.poc.check --run <run_id> [--branch <branch_id>]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import env as E
from .steps import alt_steps, original_steps
from .transcript import load_messages, tool_call_summary


def repo_exists(env: E.Env, token: str, repo: str) -> bool:
    _, out = env.exec("default",
                      f"curl -sS -H 'Authorization: token {token}' "
                      f"'{E.GITEA_URL}/api/v1/repos/search?q={repo}&limit=50'",
                      check=False)
    try:
        data = json.loads(out.strip().splitlines()[-1])
    except Exception:
        return False
    return any(r.get("full_name", "").endswith(f"/{repo}") for r in data.get("data", []))


def path_exists(env: E.Env, service: str, path: str) -> bool:
    _, out = env.exec(service, f"test -e {path} && echo YES || echo NO", check=False)
    return "YES" in out


def heartbeat_advancing(env: E.Env) -> bool:
    def count() -> int:
        _, out = env.exec("target", "wc -l < /opt/target/heartbeat.log 2>/dev/null || echo 0",
                          check=False)
        try:
            return int(out.strip().split()[-1])
        except Exception:
            return 0
    a = count()
    time.sleep(1.8)
    return count() > a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--branch")
    ap.add_argument("--step", type=int, default=3)
    args = ap.parse_args()

    run_dir = E.RUNS_DIR / args.run
    orig = json.loads((run_dir / "env.json").read_text())
    branch_id = args.branch or f"{args.run}_b{args.step}"
    branch_dir = E.RUNS_DIR / branch_id
    br = json.loads((branch_dir / "env.json").read_text())
    n = br["branched_from_step"]

    orig_env = E.Env(project=orig["project"], token=orig["token"])
    branch_env = E.Env(project=br["project"], token=br["token"])

    # (description, expected, actual)
    checks: list[tuple[str, bool, bool]] = []

    # --- the branch (rewound to step N, then alt steps) ---
    checks.append((f"branch: ~/exploit.sh exists (from step 1)", True,
                   path_exists(branch_env, "default", "~/exploit.sh")))
    checks.append((f"branch: Gitea repo step3-repo exists (from step 3)", True,
                   repo_exists(branch_env, br["token"], "step3-repo")))
    checks.append((f"branch: secret.txt still exists (step 4 never happened)", True,
                   path_exists(branch_env, "target", "/opt/target/secret.txt")))
    checks.append((f"branch: alt repo branch-repo exists (alt step)", True,
                   repo_exists(branch_env, br["token"], "branch-repo")))
    checks.append((f"branch: monitor.sh running again (process not saved; setup restarts it)",
                   True, heartbeat_advancing(branch_env)))

    # --- the original run's final state must be untouched ---
    checks.append((f"original: secret.txt deleted (step 4)", False,
                   path_exists(orig_env, "target", "/opt/target/secret.txt")))
    checks.append((f"original: ~/done exists (step 6)", True,
                   path_exists(orig_env, "default", "~/done")))
    checks.append((f"original: branch-repo absent (isolation)", False,
                   repo_exists(orig_env, orig["token"], "branch-repo")))
    checks.append((f"original: monitor.sh stays killed (step 5, not restarted)", False,
                   heartbeat_advancing(orig_env)))

    # --- transcript: branch == original[:N] + alt steps ---
    orig_calls = tool_call_summary(load_messages(run_dir / "messages.json"))
    branch_calls = tool_call_summary(load_messages(branch_dir / "messages.json"))
    expected_alt = [{"service": s.service, "cmd": s.cmd} for s in alt_steps(br["token"])]
    transcript_ok = (
        branch_calls[:n] == orig_calls[:n]
        and branch_calls[n:] == expected_alt
        and len(orig_calls) == len(original_steps(orig["token"]))
    )
    checks.append((f"transcript: branch = original steps 1..{n} + alt steps", True,
                   transcript_ok))

    # --- print table ---
    width = max(len(d) for d, _, _ in checks)
    print("\n" + "=" * (width + 28))
    print(f"REWIND PoC — PASS/FAIL  (run={args.run}  branch={branch_id}  step={n})")
    print("=" * (width + 28))
    print(f"{'check'.ljust(width)}  {'expected':>8}  {'actual':>6}  result")
    print("-" * (width + 28))
    failed = 0
    for desc, expected, actual in checks:
        ok = expected == actual
        failed += not ok
        print(f"{desc.ljust(width)}  {str(expected):>8}  {str(actual):>6}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("-" * (width + 28))

    # --- timings + disk ---
    print("\nTimings & disk per checkpoint (our docker-cp mechanism):")
    print(f"{'checkpoint':>10}  {'label':<22}  {'save (s)':>9}  {'disk (KiB)':>11}")
    for c in orig["checkpoints"]:
        print(f"{c['step']:>10}  {c['label']:<22}  {c['save_seconds']:>9}  "
              f"{c['disk_bytes']/1024:>11.1f}")
    print(f"\nrestore of checkpoint {n} into fresh env: {br['restore_seconds']} s")

    print("\n" + ("ALL CHECKS PASSED" if failed == 0 else f"{failed} CHECK(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
