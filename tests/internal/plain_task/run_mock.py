"""Run the unmodified plain task on the `rewind` sandbox with a scripted mock model.

The same as `inspect eval tests/internal/plain_task/task.py@plain_task --sandbox rewind
--model mockllm/model`, except that the mock model replays parent_mock.yaml
(Inspect's CLI can't pass scripted outputs to mockllm). Prints PARENT_LOG=<path>.

    uv run python tests/internal/plain_task/run_mock.py [--task plain_task|plain_task_basic] [--log-dir logs]
"""

import argparse
from pathlib import Path

import yaml
from inspect_ai import eval
from inspect_ai.model import get_model

from inspect_rewind import mock_outputs

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="plain_task")
ap.add_argument("--log-dir", default="logs")
args = ap.parse_args()

script = yaml.safe_load((HERE / "parent_mock.yaml").read_text())
model = get_model("mockllm/model", custom_outputs=mock_outputs(script))
log = eval(f"{HERE / 'task.py'}@{args.task}", sandbox="rewind", model=model,
           log_dir=args.log_dir)[0]
print(f"PARENT_LOG={log.location}")
