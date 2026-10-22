"""inspect_rewind: the `rewind` sandbox and `inspect-rewind fork`, on the
unmodified Inspect task in tests/internal/plain_task/.

The `Unit*` classes need no Docker. The `PlainTask*` classes run the task end to
end with the scripted `mockllm/model` (Docker required; skipped without it):

- the parent run is a normal `inspect_ai.eval(task, sandbox="rewind")`, the
  same call `inspect eval ... --sandbox rewind` makes, and one test runs the
  real `inspect eval` CLI to show the entry point is picked up;
- a fork at step N restores snapshot N with matching content hashes, starts
  with the parent's messages up to step N (plus the injected message), records
  the parent link in the sample metadata, eval metadata and tags, and loads
  with `read_eval_log`;
- the parent's log and snapshots are untouched by forking;
- a fork of a fork works, from the branch's own step and from an inherited one;
- `basic_agent()` works as well as `react()`;
- a corrupted snapshot is refused (the agent never runs).

    uv run python -m unittest tests.test_inspect_rewind -v
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.event import InfoEvent
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ToolCall, get_model

from inspect_rewind import fork, mock_outputs, snapshots
from inspect_rewind.fork import prefix_for_step
from inspect_rewind.sandbox import _outermost
from rewind.agent import normalize
from rewind.env import tar_digest
from rewind.evals import corrupt

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "tests" / "internal" / "plain_task"
TASK = "tests/internal/plain_task/task.py"


def script(name: str) -> list:
    return yaml.safe_load((EXAMPLE / name).read_text())


def docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rewind_events(sample, kind: str) -> list[dict]:
    return [e.data for e in sample.events if isinstance(e, InfoEvent) and e.source == "rewind"
            and isinstance(e.data, dict) and e.data.get("rewind") == kind]


# --- no Docker ---------------------------------------------------------------------

class UnitPrefix(unittest.TestCase):
    def call(self, i: str) -> ToolCall:
        return ToolCall(id=i, function="bash", arguments={"command": "true"})

    def test_prefix_ends_at_the_tool_result(self) -> None:
        msgs = [ChatMessageAssistant(content="", tool_calls=[self.call("a")]),
                ChatMessageTool(content="1", tool_call_id="a"),
                ChatMessageAssistant(content="", tool_calls=[self.call("b")]),
                ChatMessageTool(content="2", tool_call_id="b")]
        self.assertEqual(prefix_for_step(msgs, "a"), msgs[:2])
        self.assertEqual(prefix_for_step(msgs, "b"), msgs)
        with self.assertRaises(ValueError):
            prefix_for_step(msgs, "zzz")

    def test_step_inside_a_parallel_turn_is_refused(self) -> None:
        msgs = [ChatMessageAssistant(content="", tool_calls=[self.call("a"), self.call("b")]),
                ChatMessageTool(content="1", tool_call_id="a"),
                ChatMessageTool(content="2", tool_call_id="b")]
        with self.assertRaises(ValueError):
            prefix_for_step(msgs, "a")
        self.assertEqual(len(prefix_for_step(msgs, "b")), 3)


class UnitCapture(unittest.TestCase):
    def test_nested_and_duplicate_paths_collapse(self) -> None:
        self.assertEqual(_outermost(["/root", "/root/x", "/tmp/", "/tmp", "/rootfs"]),
                         ["/root", "/rootfs", "/tmp"])

    def test_mock_outputs(self) -> None:
        outs = mock_outputs(script("parent_mock.yaml"))
        self.assertEqual([o.message.tool_calls[0].function for o in outs],
                         ["bash", "bash", "bash", "submit"])
        self.assertIn("command", outs[0].message.tool_calls[0].arguments)


# --- end to end ----------------------------------------------------------------------

@unittest.skipUnless(docker_ok(), "needs Docker")
class PlainTaskReact(unittest.TestCase):
    task = "plain_task"

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="inspect-rewind-"))
        cls._env = os.environ.get("INSPECT_REWIND_DIR")
        os.environ["INSPECT_REWIND_DIR"] = str(cls.tmp / "snapshots")
        cls.log_dir = str(cls.tmp / "logs")
        model = get_model("mockllm/model", custom_outputs=mock_outputs(script("parent_mock.yaml")))
        run = inspect_eval(f"{TASK}@{cls.task}", sandbox="rewind", model=model,
                           log_dir=cls.log_dir, display="none")[0]
        cls.parent = read_eval_log(run.location, resolve_attachments=True)
        cls.psample = cls.parent.samples[0]
        cls.steps = snapshots(cls.psample)
        cls.parent_sha = sha(cls.parent.location)
        cls.snap_digests = {
            n: {t.name: tar_digest(t) for t in Path(e["snapshot_dir"]).glob("*.tar")}
            for n, e in cls.steps.items()}
        cls.branch, cls.report = fork(
            cls.parent.location, "chores", 2, inject="Add a fourth chore.",
            outputs=mock_outputs(script("branch_mock.yaml")))
        cls.bsample = read_eval_log(cls.branch.location, resolve_attachments=True).samples[0]

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._env is None:
            os.environ.pop("INSPECT_REWIND_DIR", None)
        else:
            os.environ["INSPECT_REWIND_DIR"] = cls._env
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # parent
    def test_parent_runs_unmodified_task_with_a_snapshot_per_tool_call(self) -> None:
        self.assertEqual(self.parent.status, "success")
        self.assertEqual(self.parent.eval.sandbox.type, "rewind")
        self.assertEqual(self.psample.scores["count_matches"].value, "C")
        self.assertEqual(sorted(self.steps), [1, 2, 3])  # 3 bash calls; submit doesn't touch it
        capture = rewind_events(self.psample, "capture")[0]["capture"]
        self.assertIn("/srv/notes", capture["default"])  # the named volume
        for e in self.steps.values():
            self.assertTrue((Path(e["snapshot_dir"]) / "meta.json").exists())
        # the snapshots differ where the agent changed something
        self.assertNotEqual(self.steps[1]["content_sha256"]["default__srv_notes.tar"],
                            self.steps[2]["content_sha256"]["default__srv_notes.tar"])

    # branch
    def test_branch_sandbox_matches_snapshot(self) -> None:
        self.assertTrue(self.report["restore_ok"])
        restore = rewind_events(self.bsample, "restore")[0]
        self.assertEqual(restore["step"], 2)
        self.assertEqual({k: v["restored"] for k, v in restore["paths"].items()},
                         self.steps[2]["content_sha256"])
        # and the agent saw the restored files: 3 chores + its own = 4
        self.assertEqual(self.bsample.scores["count_matches"].explanation.split(), ["4", "4"])

    def test_branch_transcript_is_prefix_plus_injection_plus_new_steps(self) -> None:
        n = self.report["prefix_messages"]
        self.assertTrue(self.report["prefix_ok"])
        self.assertTrue(self.report["inject_ok"])
        self.assertEqual(normalize(self.bsample.messages[:n]), normalize(self.psample.messages[:n]))
        self.assertEqual(self.bsample.messages[n].text, "Add a fourth chore.")
        self.assertEqual(self.bsample.messages[n].metadata, {"rewind": "injected"})
        new = [tc.function for m in self.bsample.messages[n + 1:]
               for tc in getattr(m, "tool_calls", None) or []]
        self.assertEqual(new[0], "bash")  # then submit (which react() strips from its messages)
        self.assertEqual(sum(m.role == "system" for m in self.bsample.messages), 1)

    def test_branch_log_carries_the_parent_link(self) -> None:
        log = read_eval_log(self.branch.location, header_only=True)
        self.assertEqual(log.status, "success")
        self.assertIn("rewind-branch", log.eval.tags)
        for link in (log.eval.metadata["rewind_branch"], self.bsample.metadata["rewind_branch"]):
            self.assertEqual(link["parent_log"], str(Path(self.parent.location).resolve()))
            self.assertEqual(link["parent_sample_id"], "chores")
            self.assertEqual(link["step"], 2)
            self.assertEqual(link["branch_model"], "mockllm/model")
            self.assertEqual(link["inject"], "Add a fourth chore.")
        self.assertEqual(log.eval.task, self.parent.eval.task)

    def test_branch_tells_agent_which_runtime_state_was_not_restored(self) -> None:
        notices = [m for m in self.bsample.messages
                   if (m.metadata or {}).get("rewind") == "restore_notice"]
        self.assertEqual(len(notices), 1)
        self.assertIn("services restarted", notices[0].text)
        self.assertIn("sessions were not restored", notices[0].text)
        self.assertEqual(notices[0].text, self.report["restore_notice"])

    def test_branch_snapshots_continue_the_parent_numbering(self) -> None:
        steps = snapshots(self.bsample)
        self.assertEqual(sorted(steps), [1, 2, 3])
        self.assertEqual(steps[2]["snapshot_dir"], self.steps[2]["snapshot_dir"])  # inherited
        self.assertNotEqual(steps[3]["snapshot_dir"], self.steps[3]["snapshot_dir"])  # its own

    def test_parent_is_untouched(self) -> None:
        self.assertEqual(sha(self.parent.location), self.parent_sha)
        for n, e in self.steps.items():
            self.assertEqual({t.name: tar_digest(t) for t in Path(e["snapshot_dir"]).glob("*.tar")},
                             self.snap_digests[n])

    def test_fork_of_a_fork(self) -> None:
        for step, want in ((3, "5"), (1, "3")):
            with self.subTest(step=step):
                outs = mock_outputs([{"bash": "echo sweep >> /srv/notes/todo.txt && "
                                              "wc -l < /srv/notes/todo.txt > /root/count.txt"},
                                     {"submit": "x"}])
                log, rep = fork(self.branch.location, "chores", step, outputs=outs,
                                model="mockllm/other")
                self.assertEqual(log.status, "success")
                self.assertTrue(rep["restore_ok"] and rep["prefix_ok"])
                s = log.samples[0]
                # step 3 of the branch has 4 chores; step 1 of the parent has 2
                self.assertEqual(s.scores["count_matches"].explanation.split(), [want, want])
                self.assertEqual(log.eval.model, "mockllm/other")

    def test_fork_inherited_step_after_model_adaptation(self) -> None:
        # Exercise the real-provider adaptation path with a local scripted
        # Model object, so this needs neither credentials nor a paid request.
        model = get_model("mockllm/adapted", custom_outputs=mock_outputs(script("branch_mock.yaml")))
        with patch("inspect_rewind.fork.is_mock", return_value=False):
            branch, report = fork(self.parent.location, "chores", 2, model=model)
        self.assertEqual(branch.status, "success")
        self.assertGreater(report["prefix_adapted"]["tool_call_ids_rewritten"], 0)
        inherited = snapshots(branch.samples[0])[1]
        prefix = prefix_for_step(branch.samples[0].messages, inherited["tool_call_id"])
        self.assertEqual(normalize(prefix), normalize(prefix_for_step(
            self.psample.messages, self.steps[1]["tool_call_id"])))
        nested, nested_report = fork(branch.location, "chores", 1,
                                     outputs=mock_outputs(script("branch_mock.yaml")))
        self.assertEqual(nested.status, "success")
        self.assertTrue(nested_report["restore_ok"] and nested_report["prefix_ok"])


@unittest.skipUnless(docker_ok(), "needs Docker")
class PlainTaskBasicAgent(PlainTaskReact):
    task = "plain_task_basic"

    def test_fork_of_a_fork(self) -> None:
        self.skipTest("covered by the react variant")


@unittest.skipUnless(docker_ok(), "needs Docker")
class PlainTaskCLI(unittest.TestCase):
    """The real command lines: `inspect eval --sandbox rewind` and `inspect-rewind fork`."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="inspect-rewind-cli-"))
        self.env = {**os.environ, "INSPECT_REWIND_DIR": str(self.tmp / "snapshots")}
        self.bin = Path(sys.executable).parent

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_ok(self, *argv: str) -> str:
        p = subprocess.run(argv, cwd=REPO, env=self.env, capture_output=True, text=True,
                           timeout=600)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p.stdout

    def test_inspect_eval_cli_picks_up_the_rewind_sandbox(self) -> None:
        # mockllm without a script only answers in text, so no tool call (and no
        # snapshot) happens; this checks `--sandbox rewind` resolves and runs.
        self.run_ok(str(self.bin / "inspect"), "eval", f"{TASK}@plain_task", "--sandbox",
                    "rewind", "--model", "mockllm/model", "--message-limit", "4",
                    "--log-dir", str(self.tmp / "logs"), "--display", "none")
        log = read_eval_log(str(next((self.tmp / "logs").glob("*.eval"))),
                            resolve_attachments=True)
        self.assertEqual(log.eval.sandbox.type, "rewind")
        self.assertEqual(rewind_events(log.samples[0], "capture")[0]["capture"]["default"],
                         ["/root", "/srv/notes", "/tmp"])

    def test_scripted_parent_then_fork_cli(self) -> None:
        out = self.run_ok(sys.executable, "tests/internal/plain_task/run_mock.py",
                          "--log-dir", str(self.tmp / "logs"))
        parent = out.split("PARENT_LOG=")[1].strip()
        out = self.run_ok(str(self.bin / "inspect-rewind"), "fork", parent, "--sample", "chores",
                          "--step", "2", "--model", "mockllm/model", "--mock-outputs",
                          "tests/internal/plain_task/branch_mock.yaml", "--inject", "Add a fourth chore.")
        branch = read_eval_log(out.split("BRANCH_LOG=")[1].strip())
        self.assertEqual(branch.status, "success")
        self.assertEqual(branch.samples[0].scores["count_matches"].explanation.split(), ["4", "4"])

    def test_corrupted_snapshot_is_refused(self) -> None:
        out = self.run_ok(sys.executable, "tests/internal/plain_task/run_mock.py",
                          "--log-dir", str(self.tmp / "logs"))
        parent = out.split("PARENT_LOG=")[1].strip()
        step = snapshots(read_eval_log(parent, resolve_attachments=True).samples[0])[2]
        corrupt(Path(step["snapshot_dir"]), {"step": 2})
        p = subprocess.run([str(self.bin / "inspect-rewind"), "fork", parent, "--sample", "chores",
                            "--step", "2", "--model", "mockllm/model", "--mock-outputs",
                            "tests/internal/plain_task/branch_mock.yaml"],
                           cwd=REPO, env=self.env, capture_output=True, text=True, timeout=600)
        self.assertEqual(p.returncode, 1, p.stderr)
        branch = read_eval_log(p.stdout.split("BRANCH_LOG=")[1].strip(), resolve_attachments=True)
        self.assertEqual(branch.status, "error")
        s = branch.samples[0]
        self.assertFalse(rewind_events(s, "restore")[0]["ok"])
        self.assertFalse(any(m.role == "assistant" for m in s.messages))  # agent never ran


if __name__ == "__main__":
    unittest.main()
