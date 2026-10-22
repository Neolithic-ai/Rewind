"""Best-effort internal probe of Inspect's native checkpointing.

This runs Inspect's development-version checkpointing against an Inspect-managed
Docker sandbox: `react` agent + mock model + `CheckpointConfig(retention="retain")`
with a per-turn trigger and the `archive` snapshot strategy (one self-contained
tar.gz per checkpoint — no restic injection into the sandbox). It then inspects
the retained checkpoint directory to show what Inspect captures per step.

The point is to answer, honestly and with real output, "does Inspect checkpointing
give us rewind-to-step-N + branch out of the box?":

  * Capture:  YES — it retains a per-step `ckpt-NNNNN.json` plus a self-contained
              sandbox archive and a host (restic) snapshot.
  * Restore:  only the *latest* committed snapshot, and only as failure-resume
              (see inspect_ai/util/_checkpoint/hydrate.py). There is no public
              "restore checkpoint N" or "branch" API.

So to rewind to an *older* step and branch we read the retained archive ourselves
(exactly what tests/internal/poc/env.py does with its own snapshots). This probe demonstrates
that the retained per-step archive is consumable: it extracts checkpoint 1's
sandbox archive and lists its contents.

This module never gates run_all.sh — it prints a report and always exits 0.
"""

from __future__ import annotations

import glob
import gzip
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path

INTERNAL = Path(__file__).resolve().parents[1]


def _write_probe_compose() -> str:
    path = INTERNAL / "env" / "compose_probe.yaml"
    path.write_text(
        "services:\n"
        "  default:\n"
        "    image: debian:bookworm-slim\n"
        '    command: ["sleep", "infinity"]\n'
        "    init: true\n"
    )
    return str(path)


def run() -> None:
    try:
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.agent import react
        from inspect_ai.dataset import Sample
        from inspect_ai.model import ModelOutput, get_model
        from inspect_ai.tool import bash
        from inspect_ai.util import (
            ArchiveSnapshots,
            CheckpointConfig,
            SandboxSnapshotConfig,
            TurnInterval,
        )
    except Exception as e:  # pragma: no cover
        print(f"[probe] Inspect import failed, skipping: {e}")
        return

    compose = _write_probe_compose()
    log_dir = tempfile.mkdtemp(prefix="rewind_probe_")
    outs = [
        ModelOutput.for_tool_call("mockllm/model", "bash", {"cmd": "echo step1 > /root/a.txt"}),
        ModelOutput.for_tool_call("mockllm/model", "bash", {"cmd": "echo step2 > /root/b.txt"}),
        ModelOutput.for_tool_call("mockllm/model", "submit", {"answer": "done"}),
    ]
    model = get_model("mockllm/model", custom_outputs=outs)
    agent = react(tools=[bash()], model=model)
    task = Task(
        dataset=[Sample(input="do it")],
        solver=agent,
        sandbox=("docker", compose),
        checkpoint=CheckpointConfig(
            trigger=TurnInterval(every=1),
            retention="retain",
            sandbox_paths={
                "default": SandboxSnapshotConfig(paths=["/root"], strategy=ArchiveSnapshots())
            },
        ),
    )

    print("[probe] running Inspect native checkpointing (react + docker + archive, retain) ...")
    try:
        logs = inspect_eval(task, model=model, log_dir=log_dir, display="none")
    except Exception as e:  # pragma: no cover
        print(f"[probe] eval raised, skipping: {e}")
        return
    log = logs[0]
    print(f"[probe] eval status: {log.status}")
    if log.status != "success":
        print(f"[probe] error: {log.error}")
        return

    ckpts = sorted(glob.glob(f"{log_dir}/*.checkpoints/**/ckpt-*.json", recursive=True))
    print(f"[probe] retained checkpoint files: {len(ckpts)}")
    sample_dir = os.path.dirname(ckpts[0]) if ckpts else None
    for p in ckpts:
        j = json.load(open(p))
        sb = (j.get("sandboxes") or {}).get("default", {})
        print(f"[probe]   {os.path.basename(p)}: trigger={j['trigger']} turn={j['turn']} "
              f"duration_ms={j.get('duration_ms')} total_bytes={j.get('size_bytes')} "
              f"sandbox_strategy={sb.get('strategy')} archive={sb.get('archive')} "
              f"archive_bytes={sb.get('size_bytes')}")

    # Prove the retained per-step archive is directly consumable (this is the
    # basis for restoring a *specific* older step, which Inspect itself won't do).
    if sample_dir:
        arch = sorted(glob.glob(f"{sample_dir}/sandboxes/default/*.tar.gz")) or \
               sorted(glob.glob(f"{sample_dir}/**/ckpt-*.tar.gz", recursive=True))
        if arch:
            raw = gzip.decompress(Path(arch[0]).read_bytes())
            with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                names = tf.getnames()[:8]
            print(f"[probe] checkpoint-1 archive {os.path.basename(arch[0])} is readable; "
                  f"members include: {names}")

    print("[probe] CONCLUSION: Inspect captures per-step retained checkpoints, but its")
    print("[probe] restore path (hydrate.py) only rehydrates the LATEST snapshot as")
    print("[probe] failure-resume — no public restore-step-N / branch API. Our PoC reads")
    print("[probe] the retained per-step snapshot directly to rewind + branch.")


if __name__ == "__main__":
    run()
