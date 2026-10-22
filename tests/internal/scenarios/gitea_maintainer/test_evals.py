"""Mock-model checks for the real-agent eval runner (`rewind.evals`).

Runs the eval pipeline with the mock model and asserts on its results.json:

  1. path1, 2 attempts, default fork points: passes; every fork point
     (reproduced, fix_committed, pr_open, mid) gets 2 branches, and every
     restore, transcript and isolation check passes.
  2. path3 (clones into /tmp/calc, as real agents do): the workspace milestones
     still fire, and branches from them restore the clone and pass.
  3. path4 (commit and push in one tool call, as real agents do): fix_committed
     still fires on that step, and a branch from it passes.
  4. negative, corrupted checkpoint: one file in the fork point's checkpoint is
     changed; the branch must be refused (content hash mismatch) and the eval
     must fail.
  5. negative, nothing to branch from: a fork point no attempt reaches is
     reported as skipped, and the eval fails because fewer than 2 branches ran.
  6. branch variants, with variants.example.yaml: every fork point is branched
     once per variant (same, alt-instruction, other-model), all checks pass;
     the injected message sits at position N and is marked; the other-model
     branch's Inspect log shows the stand-in mock `mockllm/deepseek-v3.2`
     (a differently named mock), and run.json records both models.
  7. a variant with another mock model, a system override and an inject_file
     together.

Usage (after `uv sync` and building the images, see run_all.sh):
    uv run python tests/internal/scenarios/gitea_maintainer/test_evals.py
Exits non-zero if any assertion fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from rewind.agent import MARK, load_messages  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
results: list[tuple[str, bool, str]] = []


def evals(*argv: str) -> tuple[int, dict]:
    cmd = [sys.executable, "-m", "rewind.evals", "--model", "mockllm/model",
           "--scenario", "tests/internal/scenarios/gitea_maintainer",
           "--fork-from", "reproduced,fix_committed,pr_open,mid", *argv]
    print(f"==> {' '.join(cmd[2:])}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO, stdout=subprocess.PIPE, text=True)
    print(proc.stdout, flush=True)
    out = next(line.split("=", 1)[1] for line in proc.stdout.splitlines()
               if line.startswith("EVAL_DIR="))
    return proc.returncode, json.loads((REPO / out / "results.json").read_text()) | {"dir": out}


def run_json(run_id: str) -> dict:
    return json.loads((REPO / "runs" / run_id / "run.json").read_text())


def check(desc: str, ok: bool, detail: str = "") -> None:
    results.append((desc, bool(ok), detail))


def all_checks_pass(res: dict) -> bool:
    return (all(v is True for b in res["branches"] for v in b["checks"].values())
            and all(p["ok"] is True for p in res["parent_isolation"].values()))


def main() -> int:
    rc, res = evals("--path", "path1", "--attempts", "2")
    per_point = {p["point"]: p["ran"] for p in res["fork_points"]}
    check("path1: eval passes (exit 0)", rc == 0 and res["verdict"]["ok"],
          f"rc={rc} reasons={res['verdict']['reasons']}")
    check("path1: 2 branches ran from each of reproduced, fix_committed, pr_open, mid",
          per_point == {"reproduced": 2, "fix_committed": 2, "pr_open": 2, "mid": 2},
          f"{per_point}")
    check("path1: every restore / transcript / isolation check passed", all_checks_pass(res))
    check("path1: every milestone reached by both attempts",
          all(r["reached"] == 2 for r in res["reach"].values()),
          " ".join(f"{m}={r['reached']}/{r['of']}" for m, r in res["reach"].items()))

    rc, res = evals("--path", "path3", "--attempts", "1", "--branches-per-checkpoint", "1")
    check("path3 (/tmp/calc): eval passes", rc == 0 and res["verdict"]["ok"],
          f"rc={rc} reasons={res['verdict']['reasons']}")
    check("path3: cloned, reproduced, fix_committed detected outside /root",
          all(res["reach"][m]["reached"] == 1 for m in ("cloned", "reproduced", "fix_committed")))
    fc = next(b for b in res["branches"] if b["point"] == "fix_committed")
    fp = json.loads((REPO / "runs" / fc["id"] / "run.json").read_text())["fingerprint_at_restore"]
    check("path3: branch from fix_committed restored /tmp/calc with the commit",
          all_checks_pass(res) and [c["path"] for c in fp["clones"]] == ["/tmp/calc"]
          and fc["success"], f"clones at restore: {fp['clones']}")

    rc, res = evals("--path", "path4", "--attempts", "1", "--fork-from", "fix_committed",
                    "--branches-per-checkpoint", "1", "--require-branches", "1")
    attempt = run_json(res["attempts"][0]["id"])
    at = {lbl: c["step"] for c in attempt["checkpoints"] for lbl in c["labels"]}
    check("path4 (commit + push in one call): fix_committed fires on that step, before pr_open",
          rc == 0 and res["verdict"]["ok"] and at.get("fix_committed") == 5 and at.get("pr_open") == 6
          and all_checks_pass(res) and res["branches"][0]["success"],
          f"rc={rc} milestones at {at} reasons={res['verdict']['reasons']}")

    rc, res = evals("--path", "path1", "--attempts", "1", "--fork-from", "fix_committed",
                    "--branches-per-checkpoint", "1", "--require-branches", "0",
                    "--corrupt-checkpoint")
    b = res["branches"][0]
    bad = [k for k, v in b["restore_check"]["content"]["paths"].items() if v["ok"] is False]
    check("corrupted checkpoint: branch refused on content hash mismatch",
          b.get("aborted") and b["checks"]["restore_content"] is False, f"mismatch in {bad}")
    check("corrupted checkpoint: eval fails (exit 1) on the failed check",
          rc == 1 and not res["verdict"]["ok"] and res["verdict"]["failed_checks"],
          f"rc={rc} reasons={res['verdict']['reasons']}")

    rc, res = evals("--path", "path1", "--attempts", "1", "--fork-from", "no_such_milestone")
    check("unreached fork point: reported as skipped, eval fails (fewer than 2 branches)",
          rc == 1 and res["fork_points"][0].get("skipped") and res["verdict"]["branches_ran"] == 0,
          f"rc={rc} reasons={res['verdict']['reasons']}")

    rc, res = evals("--path", "path1", "--attempts", "1", "--fork-from", "fix_committed,mid",
                    "--branches-per-checkpoint", "1",
                    "--branch-variants", "tests/internal/scenarios/gitea_maintainer/variants.example.yaml")
    grid = {(p["point"], s["variant"]): s["ran"] for p in res["fork_points"]
            for s in p["variants"]}
    check("variants: eval passes", rc == 0 and res["verdict"]["ok"],
          f"rc={rc} reasons={res['verdict']['reasons']}")
    check("variants: one branch per fork point x variant",
          grid == {(p, v): 1 for p in ("fix_committed", "mid")
                   for v in ("same", "alt-instruction", "other-model")}, f"{grid}")
    check("variants: every restore / transcript / model / injected / isolation check passed",
          all_checks_pass(res) and all(p["transcript_unchanged"]
                                       for p in res["parent_isolation"].values()))
    inject = res["variants"][1]["inject"]
    for b in res["branches"]:
        meta = run_json(b["id"])
        msgs = load_messages(REPO / "runs" / b["id"] / "messages.json")
        n = meta["prefix_messages"]
        tag = f"{b['variant']}@{b['point']}"
        if b["variant"] == "alt-instruction":
            check(f"variants: {tag}: injected user message at position N={n}, marked",
                  b["checks"]["injected"] and meta["inject_index"] == n and msgs[n].role == "user"
                  and msgs[n].text == inject and msgs[n].metadata == {MARK: "injected"}
                  and meta["inject"] == inject,
                  f"inject_index={meta['inject_index']} role={msgs[n].role}")
        else:
            check(f"variants: {tag}: nothing injected",
                  "injected" not in b["checks"] and meta["inject_index"] is None
                  and len([m for m in msgs if m.role == "user"]) == 1)
        if b["variant"] == "other-model":
            check(f"variants: {tag}: Inspect log shows the other model; run.json records both",
                  b["checks"]["model"] and b["models_called"]["inspect"] == ["mockllm/deepseek-v3.2"]
                  and meta["parent_model"] == "mockllm/model"
                  and meta["branch_model"] == "mockllm/deepseek-v3.2"
                  and meta["prefix_adapted"]["tool_call_ids_rewritten"] > 0,
                  f"called={b['models_called']} adapted={meta.get('prefix_adapted')}")
        else:
            check(f"variants: {tag}: Inspect log shows the parent's model",
                  b["models_called"]["inspect"] == ["mockllm/model"]
                  and meta["branch_model"] == meta["parent_model"] == "mockllm/model"
                  and "prefix_adapted" not in meta, f"called={b['models_called']}")
    summary = (REPO / res["dir"] / "summary.md").read_text()
    check("variants: summary has a Variant column, a row per fork point x variant",
          "| Runs | Variant |" in summary and "| Branch | Variant |" in summary
          and all(f"| `{v}` |" in summary for v in ("same", "alt-instruction", "other-model")))

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "msg.txt").write_text("Explain your plan, then continue.")
        (Path(d) / "v.yaml").write_text(
            "- name: terse-other\n  model: mockllm/other\n  system: Be terse.\n"
            "  inject_file: msg.txt\n")
        rc, res = evals("--path", "path1", "--attempts", "1", "--fork-from", "pr_open",
                        "--branches-per-checkpoint", "1", "--require-branches", "1",
                        "--branch-variants", str(Path(d) / "v.yaml"))
    b = res["branches"][0]
    meta = run_json(b["id"])
    msgs = load_messages(REPO / "runs" / b["id"] / "messages.json")
    n = meta["prefix_messages"]
    check("system + inject_file + other mock model: eval passes, all checks pass",
          rc == 0 and res["verdict"]["ok"] and all_checks_pass(res)
          and set(b["checks"]) >= {"injected", "model"}, f"rc={rc} checks={b.get('checks')}")
    check("system + inject_file: system prompt replaced and marked, message injected at N",
          msgs[0].text == "Be terse." and msgs[0].metadata == {MARK: "system_override"}
          and meta["system_override"] == "Be terse." and msgs[n].text.startswith("Explain")
          and meta["inject_file"].endswith("msg.txt") and b["models_called"]["inspect"]
          == ["mockllm/other"], f"system={msgs[0].text!r} injected={msgs[n].text!r}")

    print("\nREWIND EVALS (mock) - PASS/FAIL")
    for desc, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {desc}" + (f"\n      {detail}" if detail else ""))
    failed = sum(not ok for _, ok, _ in results)
    print("\n" + ("ALL CHECKS PASSED" if not failed else f"{failed} CHECK(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
