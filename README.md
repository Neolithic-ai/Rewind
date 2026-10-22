# Rewind

Save an agent's conversation and Docker filesystem state, then continue from
that point with the same model or a different one. Each branch gets its own
environment.

```text
Model A: start
└── Checkpoint after tool 6
    ├── Model A: original continuation
    ├── Model A: another continuation
    └── Model B: alternate continuation
        └── Checkpoint after tool 9
            ├── Model B: continue
            └── Model C: switch models again
```

## Use cases

Keep an interesting run and explore what happens next without reproducing its
entire lead-up. Compare models from the same saved state, repeat a continuation
to investigate a failure or jailbreak's reproducibility, or change an instruction
and observe how the agent responds. Branches can also help develop adversarial
test cases for monitors and control systems.

These are comparisons from a selected state, not measurements of overall task
performance or unaided model capability. Record model choices and interventions,
and keep related branches together when splitting development and evaluation data.

## Install

Requires Python 3.12+ and Docker with Compose. From this checkout:

```bash
pip install .
rewind doctor
```

## With your own agent

Use your existing Compose file and specify which service paths to capture.
Prefer dedicated Compose-managed volumes for mutable state and pin images by
digest for reproducibility. Inspect is not required.

```python
from copy import deepcopy
from rewind import Environment

with Environment.from_compose(
    "compose.yaml", capture={"default": ["/workspace"]}
) as environment:
    # Run your agent using environment.exec(service, command).
    # messages is its actual JSON-serializable conversation, including tool results.
    checkpoint = environment.checkpoint("checkpoints/step-6", messages=messages)

with checkpoint.branch() as environment:
    history = deepcopy(checkpoint.messages)
    history.append({"role": "user", "content": checkpoint.restore_notice})
    # Pass history and environment to your agent runner with the chosen model.
```

Checkpoint after a tool finishes and concurrent work has settled. Repeat
`checkpoint.branch()` to compare models; save another checkpoint inside a branch
to branch again. Your runner handles model calls, provider-specific message
formats, and provenance in optional checkpoint `metadata`. Leaving either context
removes its containers and volumes while keeping saved checkpoints.

For shell scripts:

```bash
rewind up compose.yaml --capture default:/workspace --state parent.json
rewind exec parent.json --service default -- ls -la /workspace
rewind checkpoint parent.json --output checkpoints/step-1
rewind branch checkpoints/step-1 --state branch.json
rewind down branch.json
rewind down parent.json
```

Repeat `--capture SERVICE:/PATH` for additional paths. State files are handles to
live environments; CLI checkpoints save files, while the Python API also saves
messages. Checkpoint directories must be new. Use `--help` for command options.

### Stateful services

The Python API accepts an optional `lifecycle=adapter` on
`Environment.from_compose(...)` and `checkpoint.branch(...)`. Implement these
synchronous methods on your adapter:

- `name`: a stable, versioned identifier, such as `my-service-v1`.
- `before_checkpoint(environment)`: quiesce writers and save application state
  into captured paths.
- `after_checkpoint(environment)`: resume the parent, including after partially
  failed preparation. Use bounded timeouts and make recovery safe to repeat.
- `after_restore(environment, checkpoint)`: reload application state, validate
  credentials and readiness, then release background workers. Raise on failure.

Services must start inert until restoration finishes. A failed restore hook
removes the new branch; a failed checkpoint hook publishes no checkpoint.
Checkpoints and saved environment handles record the adapter name: supply the
matching adapter explicitly when branching or calling `Environment.load`.
Export/import preserves this requirement, but does not serialize adapter code.
These hooks currently apply to the standalone Python API, not the CLI or Inspect
adapter. They reconstruct application state; they do not preserve processes.

## Branch an Inspect run

Install the optional adapter and configure your model providers and credentials:

```bash
pip install '.[inspect]'
inspect eval task.py --sandbox rewind --model "$MODEL_A"
inspect-rewind checkpoints logs/parent.eval
inspect-rewind fork logs/parent.eval --step 6 --model "$MODEL_B"
```

Use the actual log path and a listed checkpoint step. Omit `--model` to keep the
parent's model. Repeat `fork` to compare continuations, or fork a branch's log to
switch models again. Each fork prints `BRANCH_LOG=...` for that next parent.
Use `--sample SAMPLE_ID` and `--epoch` when needed. Append an instruction with
`--inject "Your instruction"` or `--inject-file FILE`. The branch log records its
parent, model, intervention, and restore verification. For OpenRouter dependencies,
install `'.[openrouter]'` instead.

Tasks need a Docker-compatible sandbox and an agent that continues from
`state.messages`. Rewind captures sandbox `exec` and `write_file` operations inside
tool calls; multiple operations update that call's checkpoint. Use sequential
tools: overlapping calls are rejected. Model switches retain assistant text and
tool calls/results, remove reasoning and other non-text assistant content, and
remap tool-call IDs. This transfers recorded context, not hidden model state.

By default, capture includes named-volume paths plus the default service's working
directory, `/root`, and `/tmp` when present. Override it per service in Compose:

```yaml
services:
  default:
    # Your existing image, command, and volumes ...
    x-rewind:
      capture: [/workspace]
```

Alternatively, set `INSPECT_REWIND_CAPTURE=default:/workspace,other-service:/data`.
Checkpoints live in `.rewind` unless `INSPECT_REWIND_DIR` is set. Keep both the
Inspect logs and checkpoint files, or export them together.

A finished run does not need live containers to branch. Earlier saved checkpoints
remain usable after completion, an error, or a time limit, provided the log
contains the matching conversation and checkpoint event. A failing or interrupted
operation may not produce a new checkpoint; use an earlier listed step. A log
alone cannot reconstruct missing environment state.

## Share a checkpoint

```bash
inspect-rewind export logs/parent.eval --step 6 --output run.rewind
inspect-rewind import run.rewind --output restored
inspect-rewind fork restored/parent.eval --step 6 --model "$MODEL_B"
```

Bundles include the selected conversation, captured files, available ancestry,
task source, and resolved Compose inputs. For tasks with local modules or data,
add `--task-dir PATH` to export. Inspect export requires a file-based task and an
explicit Compose YAML configuration. Export while source files, images, and
read-only inputs still match the run.

For standalone checkpoints, use `checkpoint.export("state.rewind")` and
`Checkpoint.import_bundle("state.rewind", "restored-state")` (import `Checkpoint`
from `rewind`), or `rewind export PATH --output state.rewind` and
`rewind import state.rewind --output restored-state`.

Imports verify content hashes without running task code or starting containers.
Images and Python dependencies are installed separately; transfer local-only
images with Docker save/load. Bundles can contain secrets from captured state.

## Limitations

Checkpointing must be enabled before the state you revisit; a transcript alone
cannot recover an uncaptured environment. Rewind restores captured files and
conversations, verifies file hashes, and restarts services. It does not preserve
process memory, background jobs, live sessions, or IP addresses. Prefer service
names and reconnect after restoring. Inspect adds a restore notice automatically;
standalone integrations should pass `checkpoint.restore_notice` to their agent.
External API effects and uncaptured state are not rolled back.

Branches require project-local storage and networks. Shared writable mounts,
external networks, fixed container names, and shared host/container namespaces
are rejected; read-only input mounts are supported. Fixed host ports can prevent
concurrent branches. Files copied over fresh images can fail exact restoration
when image-provided files were deleted or parent paths are missing; dedicated
state volumes are the simplest layout. Flush databases before capture when
needed: pausing containers does not guarantee application-level consistency.

The Inspect adapter is pinned to the tested release `0.3.279` and uses private
interfaces. Service restarts, restore notices, and transcript adaptation can all
affect continuations.

[MIT license](LICENSE).
