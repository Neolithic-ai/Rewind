"""Exercise checkpoint attribution through Inspect's real tool scheduler."""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from inspect_ai.event import ToolEvent
from inspect_ai.log._transcript import Transcript, init_transcript
from inspect_ai.model import ChatMessageAssistant, execute_tools
from inspect_ai.tool import Tool, ToolCall, tool

from inspect_rewind.sandbox import (
    ConcurrentToolCallsError,
    DockerSandboxEnvironment,
    Recorder,
    RewindSandboxEnvironment,
    current_tool_call_id,
)


class CheckpointConcurrency(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rec = Recorder(env=None, root=self.root, capture={})  # type: ignore[arg-type]
        self.rec.index()
        self.sandbox = object.__new__(RewindSandboxEnvironment)
        self.sandbox._rewind = self.rec
        self.state: dict[str, str] = {}

    async def asyncSetUp(self) -> None:
        self.transcript = Transcript()
        init_transcript(self.transcript)

    def snapshot(self, env, path: Path, meta: dict) -> dict:
        path.mkdir(parents=True)
        (path / "state.json").write_text(json.dumps(self.state))
        return {**meta, "content_sha256": {}, "save_seconds": 0, "disk_bytes": 0}

    @staticmethod
    def messages() -> list[ChatMessageAssistant]:
        return [ChatMessageAssistant(content="", tool_calls=[
            ToolCall(id=value, function="checkpoint_test_operation", arguments={"value": value})
            for value in ("a", "b")
        ])]

    async def test_serial_batch_retains_each_call_and_last_operation(self) -> None:
        async def write(file: str, contents: str) -> None:
            self.state[file] = contents
            await asyncio.sleep(0)

        @tool(name="checkpoint_test_operation", parallel=False)
        def operation() -> Tool:
            async def execute(value: str) -> str:
                """Write two files.

                Args:
                    value: Value to write.
                """
                await self.sandbox.write_file(f"{value}-1", value)
                await self.sandbox.write_file(f"{value}-2", value)
                return value
            return execute

        with patch.object(DockerSandboxEnvironment, "write_file", side_effect=write), \
                patch("inspect_rewind.sandbox.snapshot", side_effect=self.snapshot):
            await execute_tools(self.messages(), [operation()])

        self.assertEqual([e["tool_call_id"] for e in self.rec.steps.values()], ["a", "b"])
        self.assertEqual(json.loads((self.root / "step-001/state.json").read_text()),
                         {"a-1": "a", "a-2": "a"})
        self.assertEqual(json.loads((self.root / "step-002/state.json").read_text()),
                         {"a-1": "a", "a-2": "a", "b-1": "b", "b-2": "b"})

    async def test_real_parallel_tools_fail_without_publishing_a_checkpoint(self) -> None:
        async def execute_command(cmd: list[str]) -> None:
            # Keep the first call alive while the second enters its tool span.
            await asyncio.sleep(0.1)

        @tool(name="checkpoint_test_operation", parallel=True)
        def operation() -> Tool:
            async def execute(value: str) -> str:
                """Run an operation.

                Args:
                    value: Operation argument.
                """
                await self.sandbox.exec([value])
                return value
            return execute

        with patch.object(DockerSandboxEnvironment, "exec", side_effect=execute_command), \
                patch("inspect_rewind.sandbox.snapshot", side_effect=self.snapshot) as save:
            with self.assertRaisesRegex(ConcurrentToolCallsError, "concurrent tool execution"):
                await execute_tools(self.messages(), [operation()])
            save.assert_not_called()
        self.assertEqual(json.loads((self.root / "index.json").read_text())["steps"], {})
        self.assertEqual(list(self.root.glob("step-*")), [])

    async def test_both_mutating_entry_points_reject_ambiguity_before_execution(self) -> None:
        events = [ToolEvent(id=value, function="test", arguments={}, pending=True)
                  for value in ("a", "b")]
        init_transcript(Transcript(events=events))
        with patch.object(DockerSandboxEnvironment, "exec", new_callable=AsyncMock) as execute, \
                patch.object(DockerSandboxEnvironment, "write_file", new_callable=AsyncMock) as write:
            with self.assertRaises(ConcurrentToolCallsError):
                await self.sandbox.exec(["true"])
            with self.assertRaises(ConcurrentToolCallsError):
                await self.sandbox.write_file("/tmp/file", "data")
            execute.assert_not_awaited()
            write.assert_not_awaited()

    async def test_overlap_during_snapshot_preserves_previous_checkpoint(self) -> None:
        a = ToolEvent(id="a", function="test", arguments={}, pending=True)
        b = ToolEvent(id="b", function="test", arguments={}, pending=True)
        # Simulate one tool making a second sandbox operation while the next
        # tool starts. The first snapshot must survive the failed replacement.
        self.transcript._event(a)
        self.state["value"] = "original"
        with patch("inspect_rewind.sandbox.snapshot", side_effect=self.snapshot):
            await self.rec.after_op()

        started, finish = threading.Event(), threading.Event()

        def delayed_snapshot(env, path, meta):
            started.set()
            if not finish.wait(timeout=5):
                raise TimeoutError("test did not release snapshot")
            return self.snapshot(env, path, meta)

        self.state["value"] = "changed"
        with patch("inspect_rewind.sandbox.snapshot", side_effect=delayed_snapshot):
            task = asyncio.create_task(self.rec.after_op())
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                self.transcript._event(b)
                with self.assertRaises(ConcurrentToolCallsError):
                    self.rec.before_op()
                # The failure must persist even after the other call exits.
                b.pending = False
                self.transcript._event_updated(b)
            finally:
                finish.set()
            with self.assertRaises(ConcurrentToolCallsError):
                await task

        self.assertEqual(self.rec.step, 1)
        self.assertEqual(json.loads((self.root / "step-001/state.json").read_text()),
                         {"value": "original"})
        self.assertEqual(list(self.root.glob(".snapshot-*")), [])

    async def test_setup_operations_do_not_create_agent_steps(self) -> None:
        self.assertIsNone(current_tool_call_id())
        with patch("inspect_rewind.sandbox.snapshot") as save:
            await self.rec.after_op()
            save.assert_not_called()
        self.assertEqual(self.rec.step, 0)


if __name__ == "__main__":
    unittest.main()
