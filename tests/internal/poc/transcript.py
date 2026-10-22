"""Drive an Inspect task whose mock model replays a fixed list of steps.

Both the original run (tests/internal/poc/task.py) and the branch (tests/internal/poc/fork.py) use this. The
mock model (`mockllm/model`) emits one tool call per step from `custom_outputs`;
a custom solver executes each tool call against the live environment, appends the
result to the transcript, and invokes an optional per-step callback (used to fire
a checkpoint after every step).

A branch is produced by seeding the solver with the original transcript's
messages[:N] before replaying the alternative steps — so the branch transcript is
exactly the original's first N steps followed by the alternative steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolCall, tool

from .env import Env
from .steps import Step


@tool
def run():
    async def execute(service: str, cmd: str) -> str:
        """Run a shell command inside a sandbox service.

        Args:
            service: which sandbox service to run in ("default" or "target").
            cmd: the shell command to run.
        """
        return ""  # executed by the solver, not here

    return execute


def _outputs_for(steps: list[Step]) -> list[ModelOutput]:
    return [
        ModelOutput.for_tool_call("mockllm/model", "run",
                                  {"service": s.service, "cmd": s.cmd})
        for s in steps
    ]


def run_script(
    *,
    env: Env,
    steps: list[Step],
    log_dir: Path,
    seed_messages: list[ChatMessage] | None = None,
    after_step: Callable[[int, Step], None] | None = None,
) -> list[ChatMessage]:
    """Run `steps` against `env`, returning the full transcript messages."""

    @solver
    def driver():
        async def solve(state: TaskState, generate: Generate):
            if seed_messages is not None:
                state.messages = list(seed_messages)
            model = get_model()
            for i, step in enumerate(steps):
                state.output = await model.generate(state.messages, tools=state.tools)
                assistant = state.output.message
                state.messages.append(assistant)
                for tc in (assistant.tool_calls or []):
                    rc, out = env.exec(tc.arguments["service"], tc.arguments["cmd"],
                                       check=False)
                    state.messages.append(ChatMessageTool(
                        content=out.strip() or f"(rc={rc})",
                        tool_call_id=tc.id,
                        function=tc.function,
                    ))
                if after_step is not None:
                    after_step(i, step)
            return state
        return solve

    task = Task(dataset=[Sample(input="Carry out the scripted steps.")],
                solver=driver(), tools=[run()])
    logs = inspect_eval(
        task,
        model=get_model("mockllm/model", custom_outputs=_outputs_for(steps)),
        log_dir=str(log_dir),
        display="none",
    )
    log = logs[0]
    if log.status != "success":
        raise RuntimeError(f"inspect eval failed: {log.status}")
    return log.samples[0].messages


# --- (de)serialising a transcript prefix for branching ---------------------

def save_messages(messages: list[ChatMessage], path: Path) -> None:
    data = []
    for m in messages:
        entry: dict = {"role": m.role, "content": m.text}
        if isinstance(m, ChatMessageAssistant) and m.tool_calls:
            entry["tool_calls"] = [
                {"id": tc.id, "function": tc.function, "arguments": tc.arguments}
                for tc in m.tool_calls
            ]
        if isinstance(m, ChatMessageTool):
            entry["tool_call_id"] = m.tool_call_id
            entry["function"] = m.function
        data.append(entry)
    path.write_text(json.dumps(data, indent=2))


def load_messages(path: Path) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for e in json.loads(path.read_text()):
        if e["role"] == "user":
            out.append(ChatMessageUser(content=e["content"]))
        elif e["role"] == "assistant":
            tcs = [ToolCall(id=t["id"], function=t["function"], arguments=t["arguments"])
                   for t in e.get("tool_calls", [])]
            out.append(ChatMessageAssistant(content=e["content"], tool_calls=tcs or None))
        elif e["role"] == "tool":
            out.append(ChatMessageTool(content=e["content"],
                                       tool_call_id=e.get("tool_call_id"),
                                       function=e.get("function")))
    return out


def tool_call_summary(messages: list[ChatMessage]) -> list[dict]:
    """The ordered list of {service, cmd} tool calls in a transcript."""
    calls = []
    for m in messages:
        if isinstance(m, ChatMessageAssistant):
            for tc in (m.tool_calls or []):
                calls.append({"service": tc.arguments.get("service"),
                              "cmd": tc.arguments.get("cmd")})
    return calls
