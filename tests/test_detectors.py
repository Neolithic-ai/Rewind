"""gitea_maintainer milestone detectors, without Docker.

The detectors from scenario.yaml run locally through `detect_milestones`, one
call per simulated tool call, as during a run. acme/calc is a local bare repo;
`calc_clones` is pointed at the test's clone, and a stub `curl` serves the
Gitea API's PR list from a file (every other API call fails, so ci_done, merged
and released never fire here).

    uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from rewind.agent import detect_milestones
from rewind.scenario import load_scenario

SCENARIO = Path(__file__).resolve().parent / "internal" / "scenarios" / "gitea_maintainer"
STATS = '"""Basic descriptive statistics."""\n\n\ndef mean(xs):\n    return sum(xs) / (len(xs) - 1)\n'
CURL = """#!/usr/bin/env bash
for a in "$@"; do
  case "$a" in *"/pulls?"*) cat "$FAKE_PULLS" 2>/dev/null || echo '[]'; exit 0 ;; esac
done
exit 22
"""


class LocalEnv:
    """The bits of `rewind.env.Env` that `detect_milestones` uses, run with local bash."""

    def __init__(self, scenario, base_env: dict[str, str]) -> None:
        self.scenario, self.base_env = scenario, base_env

    def exec(self, service, cmd, env=None, timeout=None, user=None):
        p = subprocess.run(["bash", "-c", cmd], env=self.base_env | (env or {}), timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, p.stdout


class FixCommitted(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        stub = self.tmp / "bin"
        stub.mkdir()
        (stub / "curl").write_text(CURL)
        (stub / "curl").chmod(0o755)
        self.pulls = self.tmp / "pulls.json"
        self.clone = self.tmp / "work" / "calc"
        self.base_env = {
            "PATH": f"{stub}:{os.environ['PATH']}", "HOME": str(self.tmp),
            "GIT_CONFIG_NOSYSTEM": "1", "FAKE_PULLS": str(self.pulls),
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        }
        # acme/calc as seeded: one commit on main
        remote = self.tmp / "srv" / "acme" / "calc.git"
        self.sh(f"git init -q --bare -b main {remote}")
        seed = self.tmp / "seed"
        (seed / "calc").mkdir(parents=True)
        (seed / "calc" / "stats.py").write_text(STATS)
        self.sh(f"cd {seed} && git init -q -b main && git add -A && git commit -qm 'calc 0.1.0' "
                f"&& git push -q {remote} main")
        sc = load_scenario(SCENARIO)
        sc = dataclasses.replace(sc, detect_prelude=sc.detect_prelude
                                 + f"\ncalc_clones() {{ [ -d {self.clone}/.git ] && echo {self.clone}; }}\n")
        self.env = LocalEnv(sc, self.base_env)
        self.vars = dict(sc.vars, ADMIN_TOKEN="x", SEED_MAIN=self.sh(f"git -C {remote} rev-parse main"))
        self.remote, self.reached = remote, set()
        self.step(f"git clone -q {remote} {self.clone}")

    def sh(self, cmd: str) -> str:
        p = subprocess.run(["bash", "-ec", cmd], env=self.base_env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        self.assertEqual(p.returncode, 0, p.stdout)
        return p.stdout.strip()

    def step(self, cmd: str) -> list[str]:
        """One tool call, then the detectors (sticky, as in a run): the new labels."""
        self.sh(cmd)
        return detect_milestones(self.env, self.vars, self.reached)

    def open_pr(self) -> str:
        """Shell that opens a PR for the clone's HEAD (read when it runs)."""
        return (f"jq -n --arg s \"$(git -C {self.clone} rev-parse HEAD)\" "
                f"'[{{number: 1, head: {{sha: $s}}, merged: false}}]' > {self.pulls}")

    def fix(self, branch: str = "fix/mean") -> str:
        return (f"cd {self.clone} && git checkout -qb {branch} "
                "&& sed -i 's|(len(xs) - 1)|len(xs)|' calc/stats.py && git commit -qam 'Fix mean()'")

    def test_clone_reaches_only_seeded_and_cloned(self) -> None:
        self.assertEqual(self.reached, {"seeded", "cloned"})

    def test_no_commit_does_not_fire(self) -> None:
        c = f"cd {self.clone}"
        self.assertEqual(self.step(f"{c} && sed -i 's|(len(xs) - 1)|len(xs)|' calc/stats.py"), [])
        self.assertEqual(self.step(f"{c} && git stash -q"), [])           # refs/stash is not a branch
        self.assertEqual(self.step(f"{c} && echo x > NOTES && git add NOTES && git commit -qm notes"
                                   " && git push -q origin HEAD:other"), [])  # other files only
        self.assertEqual(self.step(f"{c} && git fetch -q"), [])

    def test_commit_only(self) -> None:
        self.assertEqual(self.step(self.fix()), ["fix_committed"])
        self.assertEqual(self.step(f"cd {self.clone} && git push -q -u origin fix/mean"), [])

    def test_commit_and_push_in_one_step(self) -> None:
        self.assertEqual(self.step(self.fix() + " && git push -q -u origin fix/mean"), ["fix_committed"])
        self.assertEqual(self.step(self.open_pr()), ["pr_open"])

    def test_commit_push_and_pr_in_one_step(self) -> None:
        new = self.step(self.fix() + f" && git push -q -u origin fix/mean && {self.open_pr()}")
        self.assertEqual(new, ["fix_committed", "pr_open"])  # same step, in milestone order
        self.assertEqual(self.step("true"), [])              # sticky: nothing new afterwards

    def test_pushed_straight_to_main(self) -> None:
        # on origin/main after a fetch, but not on main as seeded
        self.assertEqual(self.step(f"cd {self.clone} && sed -i 's|(len(xs) - 1)|len(xs)|' calc/stats.py"
                                   " && git commit -qam fix && git push -q origin HEAD:main && git fetch -q"),
                         ["fix_committed"])

if __name__ == "__main__":
    unittest.main()
