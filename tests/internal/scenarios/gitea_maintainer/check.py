"""PASS/FAIL checks for the gitea_maintainer scenario.

Expects one original run (mock path 1) and two branches made by run_all.sh:
  A: fork --checkpoint fix_committed --path path2   (the other fix)
  B: fork --checkpoint pr_open                       (rest of path 1, before CI)

Usage:
    python tests/internal/scenarios/gitea_maintainer/check.py --run <id> --branch-a <id> --branch-b <id>
Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from rewind.agent import fingerprint, load_messages, normalize, run_scorer, tool_calls  # noqa: E402
from rewind.env import RUNS_DIR, Env, tar_name  # noqa: E402
from rewind.scenario import load_scenario  # noqa: E402

MILESTONES = ["seeded", "cloned", "reproduced", "fix_committed", "pr_open", "ci_done",
              "merged", "released"]


def load(run_id: str) -> dict:
    return json.loads((RUNS_DIR / run_id / "run.json").read_text())


def ckpt_by_label(meta: dict, label: str) -> dict:
    return next(c for c in meta["checkpoints"] if label in c["labels"])


def ci_runs_in_checkpoint(run_id: str, step: int) -> list[dict]:
    """CI run history straight from a checkpoint's captured /var/lib/ci tar."""
    tar = RUNS_DIR / run_id / "checkpoints" / f"step-{step:03d}" / tar_name("ci", "/var/lib/ci")
    runs = []
    with tarfile.open(tar) as t:
        for m in t.getmembers():
            if m.name.startswith("ci/runs/") and m.name.endswith(".json"):
                runs.append(json.load(t.extractfile(m)))
    return sorted(runs, key=lambda r: r["id"])


def key(runs: list[dict]) -> list[tuple]:
    return [(r["id"], r["kind"], r["sha"]) for r in runs]


def raw_file(env: Env, variables: dict, path: str, ref: str) -> str:
    _, out = env.exec("default",
                      f'curl -sf -H "Authorization: token $ADMIN_TOKEN" '
                      f'"$GITEA_URL/api/v1/repos/acme/calc/raw/{path}?ref={ref}"',
                      env=variables)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--branch-a", required=True, help="fork from fix_committed, path2")
    ap.add_argument("--branch-b", required=True, help="fork from pr_open, path1")
    args = ap.parse_args()

    orig, br_a, br_b = load(args.run), load(args.branch_a), load(args.branch_b)
    sc = load_scenario(orig["scenario"])
    v = orig["variables"]
    env_o = Env(sc, orig["project"])
    env_a = Env(sc, br_a["project"])
    env_b = Env(sc, br_b["project"])
    checks: list[tuple[str, bool, str]] = []

    def check(desc: str, ok: bool, detail: str = "") -> None:
        checks.append((desc, bool(ok), detail))

    # --- original run (mock path 1) --------------------------------------
    live = run_scorer(env_o, v)
    check("original: mock path 1 scores success end to end",
          orig["score"].get("success") and live.get("success"),
          f"PR #{live.get('pr_number')} merge={live.get('merge_commit_sha', '')[:8]} "
          f"tag={live.get('tag_sha', '')[:8]}")
    steps = {m: next((c["step"] for c in orig["checkpoints"] if m in c["labels"]), None)
             for m in MILESTONES}
    in_order = all(steps[m] is not None for m in MILESTONES) and \
        [steps[m] for m in MILESTONES] == sorted(steps[m] for m in MILESTONES)
    check("original: all 8 milestones detected from env state, in order", in_order,
          " ".join(f"{m}@{s}" for m, s in steps.items()))
    pr_ck = ckpt_by_label(orig, "pr_open")
    ci_at_pr = ci_runs_in_checkpoint(args.run, pr_ck["step"])
    pr_runs_done = [r for r in ci_at_pr if r["kind"] == "pull_request"
                    and r["state"] in ("success", "failure")]
    check("original: pr_open checkpoint was taken before CI finished on the PR",
          not pr_runs_done and steps["ci_done"] > steps["pr_open"],
          f"CI at pr_open: {[(r['id'], r['kind'], r['state']) for r in ci_at_pr]}")

    # --- branch A: fix_committed -> path 2 (fix B) ------------------------
    check("branch A (fix_committed -> path2): scores success",
          br_a["score"].get("success") and run_scorer(env_a, v).get("success"),
          f"PR #{br_a['score'].get('pr_number')} "
          f"merge={br_a['score'].get('merge_commit_sha', '')[:8]}")
    stats_a = raw_file(env_a, v, "calc/stats.py", "v0.1.1")
    stats_o = raw_file(env_o, v, "calc/stats.py", "v0.1.1")
    check("branch A: v0.1.1 ships fix B (statistics.fmean); original ships fix A",
          "statistics.fmean(xs)" in stats_a and "sum(xs) / len(xs)" in stats_o
          and "fmean" not in stats_o)
    heads_a = [p["head"] for p in br_a["fingerprint"]["prs"]]
    heads_o = [p["head"] for p in orig["fingerprint"]["prs"]]
    check("branch A: only fix B's PR exists there; original only has fix A's PR",
          heads_a == ["fix-mean-fmean"] and heads_o == ["fix-mean-divisor"],
          f"branch={heads_a} original={heads_o}")

    # --- original untouched by both forks --------------------------------
    now = fingerprint(env_o, v)
    same = now == orig["fingerprint"]
    check("original unchanged after forks (PRs, merge commit, tag, CI history, issue)", same,
          f"tags={[(t['name'], t['sha'][:8]) for t in now['tags']]} "
          f"ci_runs={len(now['ci_runs'])}")

    # --- branch B: pr_open (before CI) -> rest of path 1 ------------------
    # A webhook can be in flight when the snapshot is taken: either already in
    # CI's on-disk queue (CI resumes it) or still undelivered in Gitea's
    # hook_task table (the restored Gitea redelivers it). Both are runs for
    # commits that existed at the checkpoint; anything for a commit created
    # later in the original (its merge commit) would be a leak.
    at_restore = br_b["fingerprint_at_restore"]
    k = len(ci_at_pr)
    existed = {at_restore["main"]} | {p["head_sha"] for p in at_restore["prs"]}
    extra = at_restore["ci_runs"][k:]
    check("branch B (pr_open): CI history at restore == original at pr_open "
          "(+ only in-flight webhooks for commits that existed then)",
          key(at_restore["ci_runs"])[:k] == key(ci_at_pr)
          and all(r["sha"] in existed for r in extra),
          f"checkpoint: {[(r['id'], r['kind'], r['state']) for r in ci_at_pr]}; "
          f"redelivered after restore: {[(r['id'], r['kind'], r['sha'][:8]) for r in extra]}")
    later_shas = {r["sha"] for r in orig["fingerprint"]["ci_runs"]} - {r["sha"] for r in ci_at_pr} - existed
    final_b = br_b["fingerprint"]["ci_runs"]
    leaked = later_shas & ({r["sha"] for r in at_restore["ci_runs"]} | {r["sha"] for r in final_b})
    check("branch B: no later CI runs leaked in (original's merge-commit run never appears)",
          later_shas and not leaked
          and key(ci_at_pr) == key(orig["fingerprint"]["ci_runs"])[:k],
          f"original-only commits={[s[:8] for s in later_shas]} leaked={len(leaked)}")
    check("branch B: final CI history = original's prefix + its own runs, no duplicates",
          key(final_b)[:k] == key(ci_at_pr) and len(final_b) > k
          and len({(r["kind"], r["sha"]) for r in final_b}) == len(final_b),
          f"{[(r['id'], r['kind'], r['sha'][:8], r['state']) for r in final_b]}")
    _, repo = env_b.exec("default", 'curl -sf -o /dev/null -w "%{http_code}" '
                         '-H "Authorization: token $ADMIN_TOKEN" "$GITEA_URL/api/v1/repos/acme/calc"',
                         env=v)
    _, integrity = env_b.exec("gitea", "sqlite3 /data/gitea/gitea.db 'PRAGMA integrity_check;'",
                              user="git")
    prs_r = at_restore["prs"]
    consistent = (repo.strip() == "200" and integrity.strip() == "ok"
                  and at_restore["issue"]["state"] == "open"
                  and [(p["head"], p["state"]) for p in prs_r] == [("fix-mean-divisor", "open")]
                  and [h["url"] for h in at_restore["hooks"]] == ["http://ci:8080/hook"])
    check("branch B: restored Gitea DB consistent (repo, issue, PR, webhook via API)",
          consistent, f"repo={repo.strip()} integrity={integrity.strip()} "
          f"issue#1={at_restore['issue']['state']} PR#{prs_r[0]['number'] if prs_r else '-'} "
          f"hooks={len(at_restore['hooks'])}")
    check("branch B: scores success", br_b["score"].get("success"))

    # --- transcripts ------------------------------------------------------
    orig_msgs = load_messages(RUNS_DIR / args.run / "messages.json")
    for name, br, path, skip in (("A", br_a, sc.mock_paths["path2"], 0),
                                 ("B", br_b, sc.mock_paths["path1"], br_b["branched_from_step"])):
        msgs = load_messages(RUNS_DIR / br["run_id"] / "messages.json")
        n = br["prefix_messages"]
        ok = (normalize(msgs[:n]) == normalize(orig_msgs[:n])
              and tool_calls(msgs[n:]) == path.steps[skip:])
        check(f"transcript {name}: = original messages[:{n}] + new steps", ok,
              f"{len(tool_calls(msgs[:n]))} inherited + {len(tool_calls(msgs[n:]))} new tool calls")

    # --- table ------------------------------------------------------------
    width = min(max(len(d) for d, _, _ in checks), 92)
    print(f"\nGITEA MAINTAINER - PASS/FAIL  (run={args.run})")
    print("=" * (width + 8))
    failed = 0
    for desc, ok, detail in checks:
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {desc}")
        if detail:
            print(f"      {detail}")
    print("=" * (width + 8))

    # --- timings & disk ---------------------------------------------------
    print("\nOriginal run: save time and disk per checkpoint (full tar snapshots)")
    print(f"{'step':>4}  {'labels':<16} {'hooks s':>7} {'paused s':>8} {'save s':>7} "
          f"{'disk KiB':>9}  {'gitea':>6} {'ci':>5} {'work':>5}")
    total = 0
    for c in orig["checkpoints"]:
        sz = c["sizes"]
        total += c["disk_bytes"]
        print(f"{c['step']:>4}  {','.join(c['labels']):<16} {c['hook_seconds']:>7} "
              f"{c['paused_seconds']:>8} {c['save_seconds']:>7} {c['disk_bytes'] / 1024:>9.0f}  "
              f"{sz[tar_name('gitea', '/data')] / 1024:>6.0f} "
              f"{sz[tar_name('ci', '/var/lib/ci')] / 1024:>5.0f} "
              f"{sz[tar_name('default', '/root')] / 1024:>5.0f}")
    print(f"total: {len(orig['checkpoints'])} checkpoints, {total / 2**20:.1f} MiB; "
          f"run wall time {orig['seconds']}s")
    for name, br in (("A", br_a), ("B", br_b)):
        bt = sum(c["disk_bytes"] for c in br["checkpoints"])
        print(f"branch {name}: restore of step {br['branched_from_step']} "
              f"({','.join(br['branched_from_labels'])}) into a fresh env: "
              f"{br['restore_seconds']} s; {len(br['checkpoints'])} new checkpoints, "
              f"{bt / 2**20:.1f} MiB; wall time {br['seconds']}s")

    print("\n" + ("ALL CHECKS PASSED" if not failed else f"{failed} CHECK(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
