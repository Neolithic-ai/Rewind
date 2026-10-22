#!/usr/bin/env bash
# Real-agent eval for the gitea_maintainer scenario: N full attempts with a real
# model, then one branch per variant (variants.example.yaml: same model,
# an injected instruction, another model) from each fork point, all scored and
# verified (restore content hash + fingerprint, transcript, injected message,
# model called, isolation). Writes
# runs/evals-<timestamp>/{results.json,summary.md} (and appends the summary to
# $GITHUB_STEP_SUMMARY when set). See `python -m rewind.evals --help`.
#
#   OPENROUTER_API_KEY=... bash tests/internal/scenarios/gitea_maintainer/real_agent.sh \
#       --model openrouter/anthropic/claude-sonnet-5 --budget-usd 10
#   bash tests/internal/scenarios/gitea_maintainer/real_agent.sh --model mockllm/model   # no key
#
# Args (passed through; later ones win over these defaults): --model M
#   (required), --attempts N (3), --fork-from a,b (here: reproduced,mid),
#   --branch-variants FILE (here: variants.example.yaml),
#   --branches-per-checkpoint K (here: 1, i.e. once per variant),
#   --require-branches B (2), --budget-usd X, --max-total-turns T, --keep
# Default spend: 3 attempts + 2 fork points x 3 variants = 6 branches, two of
# them on the ~10x cheaper other model (fewer Sonnet branches than the 8 the
# eval made before variants), well under the workflow's --budget-usd 10.
# Exits non-zero if a run crashed, a branch check failed or fewer than B
# branches ran.
set -euo pipefail

cd "$(dirname "$0")/../../../.."
export PATH="$HOME/.local/bin:$PATH"
SCENARIO=tests/internal/scenarios/gitea_maintainer
COMPOSE="$SCENARIO/compose.yaml"

echo "==> syncing deps (uv, with the openrouter extra)"
uv sync --quiet --extra openrouter

echo "==> pulling / building images"
docker pull -q gitea/gitea:1.22 >/dev/null
docker compose -p rewind_build -f "$COMPOSE" build default >/dev/null

echo "==> running evals"
exec uv run --extra openrouter python -m rewind.evals --scenario "$SCENARIO" \
    --branch-variants "$SCENARIO/variants.example.yaml" --fork-from reproduced,mid \
    --branches-per-checkpoint 1 "$@"
