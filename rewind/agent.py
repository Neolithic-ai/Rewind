"""Drive a scenario's agent through Inspect, checkpointing after every tool call.

One code path for both modes:

- mock model (default): `mockllm/model` replays a scenario mock path, one `bash`
  tool call per step, then the path's closing message. Deterministic, no key.
- real model: any Inspect model string (`openrouter/...`, `anthropic/...`); same
  tool, prompt, checkpoints and scorer. `openrouter/` models get OpenRouter's
  app attribution headers (see `rewind.openrouter`).

A custom solver runs the loop (generate -> execute each tool call in the
scenario's agent service -> append the result), so we control exactly when a
checkpoint is taken: after every tool call, with milestone labels detected from
environment state. A branch seeds the loop with the parent's messages up to the
checkpoint, so its transcript is that prefix plus the new steps.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Callable

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.event import Event
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, read_eval_log, transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import tool
from pydantic import TypeAdapter

from .env import Env, snapshot
from .openrouter import events_cost, resolve_model
from .scenario import Command, MockPath, Scenario

MOCK_MODEL = "mockllm/model"
MARK = "rewind"  # message metadata key for what rewind added to a transcript


def is_mock(model: str | None) -> bool:
    """`mockllm/<anything>` replays mock paths; differently named mock models let
    the checks tell a model switch apart without an API key."""
    return model is None or model.startswith("mockllm/")


@tool
def bash():
    async def execute(cmd: str) -> str:
        """Run a bash command on your workstation and return its output.

        Args:
            cmd: The bash command to run.
        """
        return ""  # executed by the solver against the live environment

    return execute


# --- environment-side helpers ----------------------------------------------

def detect_milestones(env: Env, variables: dict[str, str], reached: set[str]) -> list[str]:
    """Run the detectors for milestones not reached yet; return the new ones."""
    new = []
    for m in env.scenario.milestones:
        if m.name in reached:
            continue
        # pipefail: a failing `curl ... | jq -e` must not count as "reached".
        rc, _ = env.exec(m.service, f"set -o pipefail\n{env.scenario.detect_prelude}\n{m.detect}",
                         env=variables, timeout=30)
        if rc == 0:
            reached.add(m.name)
            new.append(m.name)
    return new


def _json_command(env: Env, cmd: Command, variables: dict[str, str], timeout: float) -> dict:
    """Run a scenario command and parse the JSON object it prints."""
    rc, out = env.exec(cmd.service, cmd.source(env.scenario.root), env=variables,
                       user=cmd.user, timeout=timeout)
    out = out.strip()
    try:
        return json.loads(out[out.index("{"):])
    except ValueError:
        return {"success": False, "error": f"rc={rc}: {out[-2000:]}"}


def run_scorer(env: Env, variables: dict[str, str]) -> dict:
    return _json_command(env, env.scenario.scorer, variables, timeout=300)


def fingerprint(env: Env, variables: dict[str, str]) -> dict | None:
    fp = env.scenario.fingerprint
    return _json_command(env, fp, variables, timeout=60) if fp else None


def fingerprint_diff(a: dict | None, b: dict | None, volatile: list[str]) -> list[str] | None:
    """Fingerprint keys whose values differ, leaving out `volatile` ones (state
    background services may still change on their own). None if either is missing."""
    if a is None or b is None:
        return None
    return sorted(k for k in (set(a) | set(b)) - set(volatile) if a.get(k) != b.get(k))


def initial_messages(scenario: Scenario) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    if scenario.system:
        msgs.append(ChatMessageSystem(content=scenario.system))
    msgs.append(ChatMessageUser(content=scenario.prompt))
    return msgs


def mock_outputs(path: MockPath, skip: int = 0, model: str = MOCK_MODEL) -> list[ModelOutput]:
    outs = [ModelOutput.for_tool_call(model, "bash", {"cmd": cmd})
            for cmd in path.steps[skip:]]
    outs.append(ModelOutput.from_content(model, path.final))
    return outs


# --- branch variants ----------------------------------------------------------

def branch_prefix(prefix: list[ChatMessage], system: str | None = None,
                  inject: str | None = None) -> tuple[list[ChatMessage], int | None]:
    """A branch's seed transcript: the parent's prefix, with the system prompt
    replaced (or added) if `system` is given and a user message appended if
    `inject` is. Both are marked in their metadata (`rewind: system_override` /
    `injected`), so analysis can tell them from what the agent saw or did.
    Returns (messages, index of the injected message or None)."""
    msgs = list(prefix)
    if system is not None:
        sysmsg = ChatMessageSystem(content=system, metadata={MARK: "system_override"})
        i = next((i for i, m in enumerate(msgs) if m.role == "system"), None)
        if i is None:
            msgs.insert(0, sysmsg)
        else:
            msgs[i] = sysmsg
    if inject is None:
        return msgs, None
    msgs.append(ChatMessageUser(content=inject, metadata={MARK: "injected"}))
    return msgs, len(msgs) - 1


def adapt_for_model(messages: list[ChatMessage]) -> tuple[list[ChatMessage], dict]:
    """Make a transcript written by one model safe to replay to another provider.

    - Reasoning/thinking blocks are dropped: they carry provider signatures
      (Anthropic thinking, OpenAI/OpenRouter encrypted reasoning) that another
      provider rejects or cannot use.
    - Other non-text content in assistant turns (server-side tool use, data
      blocks) is dropped; text blocks lose provider `internal` state and
      citations, and all-text content becomes a plain string.
    - Tool-call ids are renumbered `call00001`, `call00002`, ...: 9 alphanumeric
      characters, a format every provider accepts (Anthropic wants
      `^[a-zA-Z0-9_-]+$`, Mistral exactly 9 alphanumerics), and tool results are
      pointed at the new ids.
    Returns (messages, counts of what changed)."""
    ids: dict[str, str] = {}
    stats = {"tool_call_ids_rewritten": 0, "reasoning_blocks_dropped": 0,
             "other_blocks_dropped": 0}

    def new_id(old: str) -> str:
        if old not in ids:
            ids[old] = f"call{len(ids) + 1:05d}"
            stats["tool_call_ids_rewritten"] += 1
        return ids[old]

    def text_only(content, assistant: bool):
        if isinstance(content, str):
            return content
        kept = []
        for c in content:
            if isinstance(c, ContentReasoning):
                stats["reasoning_blocks_dropped"] += 1
            elif isinstance(c, ContentText):
                kept.append(ContentText(text=c.text, refusal=c.refusal))
            elif assistant:
                stats["other_blocks_dropped"] += 1
            else:
                kept.append(c)  # images etc. in user/tool turns are portable
        if all(isinstance(c, ContentText) for c in kept):
            return "\n".join(c.text for c in kept)
        return kept

    out: list[ChatMessage] = []
    for m in messages:
        if isinstance(m, ChatMessageAssistant):
            calls = [dataclasses.replace(tc, id=new_id(tc.id)) for tc in m.tool_calls or []]
            m = m.model_copy(update={"content": text_only(m.content, True),
                                     "tool_calls": calls or None})
        elif isinstance(m, ChatMessageTool):
            m = m.model_copy(update={"content": text_only(m.content, False),
                                     "tool_call_id": new_id(m.tool_call_id)
                                     if m.tool_call_id else None})
        out.append(m)
    return out, stats


def logged_models(log_dir: Path) -> dict[str, list[str]]:
    """Which models a run's Inspect log shows were called: the model Inspect
    called (`inspect`) and the model the API said served it (`served`)."""
    called: list[str] = []
    served: list[str] = []
    for f in sorted(log_dir.glob("*.eval")):
        for sample in read_eval_log(f).samples or []:
            for e in sample.events:
                if isinstance(e, ModelEvent):
                    if e.model not in called:
                        called.append(e.model)
                    if e.output.model and e.output.model not in served:
                        served.append(e.output.model)
    return {"inspect": called, "served": served}


# --- the run loop -------------------------------------------------------------

def drive(*, env: Env, variables: dict[str, str], out_dir: Path,
          model: str | None, outputs: list[ModelOutput] | None,
          seed_messages: list[ChatMessage] | None, start_step: int,
          reached: set[str], log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
          max_turns: int | None = None, budget_usd: float | None = None,
          note: dict | None = None) -> tuple[list[ChatMessage], list[dict], dict, dict]:
    """Run the agent; checkpoint after each tool call into out_dir/checkpoints.

    `max_turns` can lower (never raise) the scenario's `agent.max_turns`.
    `budget_usd` stops the agent once its model calls have cost that much (when
    the provider or Inspect reports a cost; OpenRouter does). `note` is logged
    as an info event at the start of the Inspect transcript (a branch records
    what it changed there).
    Returns (messages, checkpoint metas, scorer result, usage).
    """
    sc = env.scenario
    agent = sc.agent
    ckpts: list[dict] = []
    score_result: dict = {}
    stopped: dict = {}
    step = start_step
    turns = int(agent.get("max_turns", 60))
    if max_turns is not None:
        turns = max(0, min(turns, max_turns))

    def after_tool_call(messages: list[ChatMessage], cmd: str, rc: int) -> None:
        nonlocal step
        step += 1
        labels = detect_milestones(env, variables, reached)
        meta = snapshot(env, out_dir / "checkpoints" / f"step-{step:03d}", {
            "step": step, "labels": labels, "reached": sorted(reached),
            "n_messages": len(messages),
            "cmd": cmd, "rc": rc,
            "fingerprint": fingerprint(env, variables),  # what a restore must reproduce
        })
        ckpts.append(meta)
        tag = f" [{', '.join(labels)}]" if labels else ""
        log(f"  step {step:>2}{tag}: rc={rc} save={meta['save_seconds']}s "
            f"disk={meta['disk_bytes'] / 1024:.0f}KiB  {cmd.splitlines()[0][:70]}")

    @solver
    def loop():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.messages = list(seed_messages) if seed_messages else initial_messages(sc)
            if note:
                transcript().info(note, source="rewind")
            m = get_model()
            tools = [bash()]
            for _ in range(turns):
                state.output = await m.generate(
                    state.messages, tools=tools, config=GenerateConfig(parallel_tool_calls=False))
                state.messages.append(state.output.message)
                calls = state.output.message.tool_calls or []
                if not calls:
                    break
                for tc in calls:
                    cmd = str(tc.arguments.get("cmd", ""))
                    if tc.function != "bash":
                        rc, out = 127, f"unknown tool {tc.function!r}"
                    else:
                        rc, out = env.exec(agent["service"], cmd, workdir=agent.get("workdir"),
                                           timeout=float(agent.get("tool_timeout", 300)))
                    state.messages.append(ChatMessageTool(
                        content=f"{out.rstrip()}\n[exit code {rc}]".lstrip(),
                        tool_call_id=tc.id, function=tc.function))
                    after_tool_call(state.messages, cmd, rc)
                # checked after the tool calls, so the transcript stays well formed
                spent = events_cost(transcript().events) if budget_usd is not None else None
                if spent is not None and spent >= budget_usd:
                    log(f"  budget reached: ${spent:.4f} >= ${budget_usd:.4f}; stopping the agent")
                    stopped["budget_exhausted"] = True
                    break
            return state
        return solve

    @scorer(metrics=[accuracy()])
    def env_scorer():
        async def score(state: TaskState, target: Target) -> Score:
            res = run_scorer(env, variables)
            score_result.update(res)
            return Score(value=CORRECT if res.get("success") else INCORRECT,
                         explanation=json.dumps(res), metadata=res)
        return score

    if is_mock(model):
        inspect_model = get_model(model or MOCK_MODEL, custom_outputs=outputs or [])
    else:
        inspect_model = resolve_model(model)
    task = Task(dataset=[Sample(input=sc.prompt)], solver=loop(), scorer=env_scorer(),
                name=sc.name)
    logs = inspect_eval(task, model=inspect_model, log_dir=str(out_dir / "log"),
                        display="none")
    if logs[0].status != "success":
        raise RuntimeError(f"inspect eval failed: {logs[0].status} {logs[0].error}")
    sample = logs[0].samples[0]
    return sample.messages, ckpts, score_result, usage(logs[0], sample.events) | stopped


def usage(log: EvalLog, events: list[Event]) -> dict:
    """Token totals, plus the USD cost when the provider or Inspect reports one."""
    tot = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for u in log.stats.model_usage.values():
        tot["input_tokens"] += u.input_tokens
        tot["output_tokens"] += u.output_tokens
        tot["total_tokens"] += u.total_tokens
    tot["cost_usd"] = events_cost(events)
    return tot


# --- transcript (de)serialisation ------------------------------------------

_MESSAGES = TypeAdapter(list[ChatMessage])


def save_messages(messages: list[ChatMessage], path: Path) -> None:
    path.write_text(_MESSAGES.dump_json(messages, indent=2).decode())


def load_messages(path: Path) -> list[ChatMessage]:
    return _MESSAGES.validate_json(path.read_text())


def normalize(messages: list[ChatMessage]) -> list[dict]:
    """Comparable view of a transcript: role, text, tool calls, tool results.

    Tool-call ids are replaced by their order of first appearance, so a prefix
    whose ids were renumbered for another provider (`adapt_for_model`) still
    compares equal, while a tool result pointing at the wrong call does not."""
    out = []
    ids: dict[str | None, int] = {}
    for m in messages:
        e: dict = {"role": m.role, "text": m.text}
        if getattr(m, "tool_calls", None):
            e["tool_calls"] = [(tc.function, tc.arguments, ids.setdefault(tc.id, len(ids)))
                               for tc in m.tool_calls]
        if m.role == "tool":
            e["tool_call_id"] = ids.setdefault(m.tool_call_id, len(ids))
        out.append(e)
    return out


def tool_calls(messages: list[ChatMessage]) -> list[str]:
    return [str(tc.arguments.get("cmd")) for m in messages
            for tc in (getattr(m, "tool_calls", None) or [])]
