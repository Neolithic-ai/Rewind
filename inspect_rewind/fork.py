"""Fork a sample of an `.eval` log at step N into a new, normal `.eval` log.

A branch is a fresh run of the *same task* (loaded from the parent log's task
file and task args) on the same sample, with one solver put in front of the
task's own solver chain (`rewind_seed`):

1. it restores the parent's snapshot N into the sample's `rewind` sandbox and
   verifies every path's content hash (a mismatch fails the sample: the agent
   never runs on a wrong environment);
2. it replaces `state.messages` with the parent's messages up to and including
   the tool result of step N (plus `--inject` text as a new user message).

Then the task's own solver runs as usual. System messages are the exception:
the seed keeps the ones already in `state.messages` (from the sample input or
setup solvers) and drops the parent's, because agents like `react()` and
`basic_agent()` insert their system prompt themselves when they start. An agent
that ignores the incoming `state.messages` (builds its own conversation) does
not continue the parent; `fork` detects that and reports `prefix_ok: False`.

Sample setup (files, setup script) and the task's `setup` solvers run before
the seed, on the fresh sandbox; the restore then overwrites the captured paths.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai import Task, task_with
from inspect_ai import eval as inspect_eval
from inspect_ai._eval.loader import load_tasks
from inspect_ai.event import InfoEvent
from inspect_ai.log import EvalLog, EvalSample, read_eval_log, transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import Generate, TaskState, chain, solver
from inspect_ai.util import sandbox

from rewind.agent import adapt_for_model, is_mock, normalize
from rewind.openrouter import resolve_model

from .sandbox import SOURCE, RewindSandboxEnvironment

MARK = "rewind"  # message metadata key, as in rewind/


@dataclass
class Seed:
    snapshot_dir: Path
    step: int
    inherited: dict[int, dict]
    messages: list[ChatMessage]
    inject: str | None
    note: dict


_SEEDS: dict[str, Seed] = {}  # branch id -> seed (keeps solver args loggable)


def restore_notice(step: int) -> str:
    return (f"<rewind_notice>Resumed from checkpoint {step}. Captured files were restored "
            "and services restarted. Process memory, background jobs, and live shell or "
            "network sessions were not restored. Network addresses may differ. Verify "
            "endpoints and reconnect any sessions you need, then continue from the latest "
            "tool result.</rewind_notice>")


@solver
def rewind_seed(branch: str):
    """Restore snapshot N and seed the conversation; see the module docstring."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        seed = _SEEDS[branch]
        transcript().info({"rewind": "branch", **seed.note}, source=SOURCE)
        env = sandbox().as_type(RewindSandboxEnvironment)
        check = await env.restore(seed.snapshot_dir, seed.step, seed.inherited)
        transcript().info({"rewind": "restore", "step": seed.step,
                           "snapshot_dir": str(seed.snapshot_dir), **check}, source=SOURCE)
        if not check["ok"]:
            bad = [k for k, v in check["paths"].items() if not v["ok"]]
            raise RuntimeError(f"restored sandbox does not match snapshot {seed.step}: {bad}")
        system = [m for m in state.messages if m.role == "system"]
        msgs = system + [m for m in seed.messages if m.role != "system"]
        if seed.inject is not None:
            msgs.append(ChatMessageUser(content=seed.inject, metadata={MARK: "injected"}))
        msgs.append(ChatMessageUser(content=restore_notice(seed.step),
                                    metadata={MARK: "restore_notice"}))
        state.messages = msgs
        return state

    return solve


# --- reading the parent -----------------------------------------------------------

def find_sample(log: EvalLog, sample_id: str | int | None = None, epoch: int = 1) -> EvalSample:
    if sample_id is None:
        candidates = [s for s in log.samples or [] if s.epoch == epoch]
        if len(candidates) == 1:
            return candidates[0]
        ids = ", ".join(str(s.id) for s in candidates) or "none"
        raise ValueError(f"choose --sample for epoch {epoch}; available samples: {ids}")
    for s in log.samples or []:
        if str(s.id) == str(sample_id) and s.epoch == epoch:
            return s
    raise ValueError(f"sample {sample_id!r} (epoch {epoch}) not in {log.location}")


def snapshots(sample: EvalSample) -> dict[int, dict]:
    """step -> snapshot entry, from the sample's rewind info events (read the log
    with `resolve_attachments=True`). A branch's log also lists the parent steps
    it inherited (in its branch event)."""
    out: dict[int, dict] = {}
    for e in sample.events:
        if not (isinstance(e, InfoEvent) and e.source == SOURCE and isinstance(e.data, dict)):
            continue
        if e.data.get("rewind") == "branch":
            out.update({int(k): v for k, v in (e.data.get("inherited") or {}).items()})
        elif e.data.get("rewind") == "snapshot":
            out[int(e.data["step"])] = {k: v for k, v in e.data.items() if k != "rewind"}
    return out


def prefix_for_step(messages: list[ChatMessage], tool_call_id: str) -> list[ChatMessage]:
    """The messages up to and including the tool result of `tool_call_id`.

    Refuses a step in the middle of a turn whose other tool calls have results
    after it (parallel calls): the prefix would hold a call with no result."""
    idx = next((i for i, m in enumerate(messages)
                if isinstance(m, ChatMessageTool) and m.tool_call_id == tool_call_id), None)
    if idx is None:
        raise ValueError(f"no tool result for call {tool_call_id!r} in the sample's messages "
                         "(compacted or trimmed by the agent?)")
    turn = next(m for m in reversed(messages[:idx]) if isinstance(m, ChatMessageAssistant))
    done = {m.tool_call_id for m in messages[:idx + 1] if isinstance(m, ChatMessageTool)}
    missing = [tc.id for tc in turn.tool_calls or [] if tc.id not in done]
    if missing:
        raise ValueError(f"tool call {tool_call_id!r} shares a turn with calls {missing} whose "
                         "results come later; fork from the turn's last step instead")
    return list(messages[:idx + 1])


def mock_outputs(spec: list[Any], model: str = "mockllm/model") -> list[ModelOutput]:
    """Scripted `mockllm` outputs for Inspect's own tools: each item is `{"bash": cmd}`, `{"submit": answer}`,
    `{"tool": name, "args": {...}}` or `{"content": text}`."""
    if not isinstance(spec, list):
        raise ValueError("mock outputs must be a list of output objects")
    outs = []
    for index, item in enumerate(spec, 1):
        if (not isinstance(item, dict)
                or sum(key in item for key in ("bash", "submit", "tool", "content")) != 1):
            raise ValueError(f"mock output {index} must specify exactly one of bash, submit, tool, content")
        if "bash" in item:
            outs.append(ModelOutput.for_tool_call(model, "bash", {"command": item["bash"]}))
        elif "submit" in item:
            outs.append(ModelOutput.for_tool_call(model, "submit", {"answer": item["submit"]}))
        elif "tool" in item:
            outs.append(ModelOutput.for_tool_call(model, item["tool"], item.get("args") or {}))
        else:
            outs.append(ModelOutput.from_content(model, item["content"]))
    return outs


def _task_file(log: EvalLog) -> Path | None:
    if not log.eval.task_file:
        return None
    path = Path(log.eval.task_file).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, *(
        parent / path for parent in Path(log.location).resolve().parents)]
    found = next((p.resolve() for p in candidates if p.is_file()), None)
    if found is None:
        raise FileNotFoundError(f"parent task file {str(path)!r} is unavailable; "
                               "use --task /path/to/task.py@task_name after moving a run")
    return found


def _load_task(log: EvalLog, task_spec: str | None = None) -> Task:
    name = log.eval.task_registry_name or log.eval.task
    spec = task_spec or name
    if task_spec is None and (path := _task_file(log)) is not None:
        spec = f"{path}@{name.split('/')[-1]}"
    tasks = load_tasks([spec], dict(log.eval.task_args or {}))
    if len(tasks) != 1:
        raise ValueError(f"could not load task {spec!r} from the parent log")
    return tasks[0]


# --- fork --------------------------------------------------------------------------

def fork(log_file: str | Path, sample_id: str | int | None, step: int, *, epoch: int = 1,
         model: str | Model | None = None, inject: str | None = None,
         outputs: list[ModelOutput] | None = None, log_dir: str | None = None,
         display: str = "none", task_spec: str | None = None,
         snapshot_dir: str | Path | None = None) -> tuple[EvalLog, dict]:
    """Branch `sample_id` of `log_file` at `step`; return (branch log, report).

    `model` defaults to the parent's. `outputs` are scripted outputs for a
    `mockllm/...` model. The report says whether the branch's transcript starts
    with the parent's prefix (`prefix_ok`) and what the restore verified.
    `sample_id=None` selects the only sample in the requested epoch. Explicit
    `task_spec` and `snapshot_dir` allow copied artifacts to be used elsewhere.
    """
    # resolve_attachments: Inspect moves long strings in events and messages
    # (snapshot paths, prompts) into attachments
    parent = read_eval_log(str(log_file), resolve_attachments=True)
    sample = find_sample(parent, sample_id, epoch)
    steps = snapshots(sample)
    if step not in steps:
        raise ValueError(f"no snapshot for step {step}; this sample has steps {sorted(steps)}")
    entry = dict(steps[step])
    if snapshot_dir is not None:
        entry["snapshot_dir"] = str(Path(snapshot_dir).expanduser().resolve())
    checkpoint = Path(entry["snapshot_dir"])
    if not (checkpoint / "meta.json").is_file():
        raise FileNotFoundError(f"checkpoint files are unavailable at {checkpoint}; "
                                "copy the checkpoint directory and pass --snapshot-dir PATH")
    checkpoint_meta = json.loads((checkpoint / "meta.json").read_text())
    if not isinstance(checkpoint_meta, dict):
        raise ValueError(f"checkpoint metadata at {checkpoint} must be a JSON object")
    if checkpoint_meta.get("content_sha256") != entry.get("content_sha256"):
        raise ValueError(f"checkpoint at {checkpoint} does not match the saved hashes for step {step}; "
                         "select that step's checkpoint directory")
    prefix = prefix_for_step(sample.messages, entry["tool_call_id"])
    inherited = {k: dict(v) for k, v in steps.items() if k <= step}
    inherited[step] = dict(entry)
    task = _load_task(parent, task_spec)
    original = next((s for s in task.dataset if str(s.id) == str(sample.id)), None)
    if original is None:
        raise ValueError(f"sample {sample.id!r} is no longer in the task's dataset")

    parent_model = parent.eval.model
    model_name = model.name if isinstance(model, Model) else (model or parent_model)
    adapted = None
    if not is_mock(model_name) and model_name != parent_model:
        original_ids = [tc.id for m in prefix for tc in getattr(m, "tool_calls", None) or []]
        prefix, adapted = adapt_for_model(prefix)
        adapted_ids = [tc.id for m in prefix for tc in getattr(m, "tool_calls", None) or []]
        ids = dict(zip(original_ids, adapted_ids, strict=True))
        # Inherited snapshots must refer to the IDs in the branch transcript,
        # otherwise a later fork cannot locate their tool results.
        for snapshot_entry in inherited.values():
            snapshot_entry["tool_call_id"] = ids[snapshot_entry["tool_call_id"]]
    if isinstance(model, Model):
        branch_model = model
    elif is_mock(model_name):
        branch_model = get_model(model_name, custom_outputs=outputs or [])
    else:
        branch_model = resolve_model(model_name)

    branch_id = uuid.uuid4().hex[:12]
    link = {
        "branch_id": branch_id,
        "parent_log": str(Path(parent.location).resolve()),
        "parent_eval_id": parent.eval.eval_id,
        "parent_sample_id": sample.id,
        "parent_sample_uuid": sample.uuid,
        "parent_epoch": epoch,
        "step": step,
        "tool_call_id": entry["tool_call_id"],
        "snapshot_dir": entry["snapshot_dir"],
        "prefix_messages": len(prefix),
        "parent_model": parent_model,
        "branch_model": model_name,
        "inject": inject,
        "restore_notice": restore_notice(step),
        "prefix_adapted": adapted,
    }
    sb = sample.sandbox if sample.sandbox and sample.sandbox.config is not None else parent.eval.sandbox
    update = {"metadata": {**(original.metadata or {}), "rewind_branch": link}}
    if task_spec is None and sb is not None:
        # Sample-level configs outrank task overrides in Inspect. Use the logged
        # effective config, including a bundle's relocated Compose file. Logs
        # may retain task-relative filenames; Sample configs are resolved from
        # the caller's cwd, so make these absolute before installing an override.
        if isinstance(sb.config, str) and not Path(sb.config).is_absolute():
            task_file = _task_file(parent)
            base = task_file.parent if task_file is not None else Path.cwd()
            sb = sb.model_copy(update={"config": str((base / sb.config).resolve())})
        update["sandbox"] = sb.model_copy(update={"type": "rewind"})
    original = original.model_copy(update=update)
    task = task_with(task, dataset=[original], epochs=1,
                     solver=chain(rewind_seed(branch_id), task.solver))
    _SEEDS[branch_id] = Seed(checkpoint, step, inherited, prefix, inject,
                             {**link, "inherited": {str(k): v for k, v in inherited.items()}})
    try:
        logs = inspect_eval(
            task, model=branch_model,
            sandbox=("rewind", sb.config) if sb is not None and task_spec is None else "rewind",
            log_dir=log_dir or str(Path(parent.location).parent),
            tags=[*(parent.eval.tags or []), "rewind-branch"],
            metadata={**(parent.eval.metadata or {}), "rewind_branch": link},
            display=display,  # type: ignore[arg-type]
            **_generate_config(parent))
    finally:
        _SEEDS.pop(branch_id, None)
    log = read_eval_log(logs[0].location, resolve_attachments=True)
    report: dict[str, Any] = {"branch_log": log.location, "status": log.status, **link}
    if log.samples:
        bs = log.samples[0]
        report["prefix_ok"] = normalize(bs.messages[:len(prefix)]) == normalize(prefix)
        if inject is not None:
            m = bs.messages[len(prefix)] if len(bs.messages) > len(prefix) else None
            report["inject_ok"] = bool(m and m.role == "user" and m.text == inject
                                       and (m.metadata or {}).get(MARK) == "injected")
        restore = next((e.data for e in bs.events if isinstance(e, InfoEvent)
                        and e.source == SOURCE and isinstance(e.data, dict)
                        and e.data.get("rewind") == "restore"), None)
        report["restore_ok"] = restore["ok"] if restore else None
    return log, report


def _generate_config(parent: EvalLog) -> dict:
    cfg = parent.plan.config if parent.plan else GenerateConfig()
    return cfg.model_dump(exclude_none=True)
