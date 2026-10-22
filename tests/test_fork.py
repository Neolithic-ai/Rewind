"""Checkpoint lookup must stay within a branch's actual history."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rewind.fork import resolve_checkpoint


class CheckpointLineage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patch = patch("rewind.fork.RUNS_DIR", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.write_run("parent", [(1, "start"), (2, "fork"), (3, "parent_only"),
                                  (4, "parent_future")])
        self.write_run("branch", [(3, "branch_only")], "parent", 2)
        self.write_run("grandchild", [], "branch", 1)

    def write_run(self, name, checkpoints, parent=None, step=None):
        directory = self.root / name
        directory.mkdir()
        (directory / "run.json").write_text(json.dumps({
            "checkpoints": [{"step": n, "labels": [label]} for n, label in checkpoints],
            "parent_run": parent, "branched_from_step": step,
        }))

    def test_own_checkpoint_wins_over_parent(self):
        path, meta = resolve_checkpoint(self.root / "branch", "3")
        self.assertEqual(path, self.root / "branch/checkpoints/step-003")
        self.assertEqual(meta["labels"], ["branch_only"])

    def test_inherited_checkpoint_is_available_by_step_and_label(self):
        for which in ("2", "fork"):
            path, _ = resolve_checkpoint(self.root / "branch", which)
            self.assertEqual(path, self.root / "parent/checkpoints/step-002")

    def test_parent_future_is_not_part_of_branch_history(self):
        for which in ("4", "parent_only", "parent_future"):
            with self.subTest(which=which), self.assertRaises(SystemExit):
                resolve_checkpoint(self.root / "branch", which)

    def test_each_ancestor_is_limited_by_all_intervening_forks(self):
        path, _ = resolve_checkpoint(self.root / "grandchild", "start")
        self.assertEqual(path, self.root / "parent/checkpoints/step-001")
        for which in ("2", "fork", "3", "branch_only", "4"):
            with self.subTest(which=which), self.assertRaises(SystemExit):
                resolve_checkpoint(self.root / "grandchild", which)
