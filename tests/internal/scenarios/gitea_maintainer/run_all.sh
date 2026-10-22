#!/usr/bin/env bash
# End to end for the gitea_maintainer scenario: run mock path 1 (checkpoint after
# every tool call), fork from `fix_committed` with mock path 2 and from `pr_open`
# (before CI) with the rest of path 1, then print the PASS/FAIL checks and the
# timing/disk report. Exits non-zero on any FAIL.
#
#   bash tests/internal/scenarios/gitea_maintainer/run_all.sh
#
# Env:
#   KEEP=1   leave the Docker environments running afterwards
set -euo pipefail

cd "$(dirname "$0")/../../../.."
export PATH="$HOME/.local/bin:$PATH"
SCENARIO=tests/internal/scenarios/gitea_maintainer
COMPOSE="$SCENARIO/compose.yaml"

echo "==> syncing deps (uv)"
uv sync --quiet

echo "==> pulling / building images"
docker pull -q gitea/gitea:1.22 >/dev/null
docker compose -p rewind_build -f "$COMPOSE" build default >/dev/null

PROJECTS=()
cleanup() {
  if [ "${KEEP:-0}" != "1" ]; then
    echo
    echo "==> tearing down environments"
    for p in "${PROJECTS[@]}"; do
      docker compose -p "$p" -f "$COMPOSE" down -v -t2 >/dev/null 2>&1 || true
    done
  fi
}
trap cleanup EXIT
project_of() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["project"])' "runs/$1/run.json"; }

echo "==> original run: mock path 1 (fix A)"
RUN_ID="$(uv run python -m rewind.run --scenario "$SCENARIO" --path path1 | sed -n 's/^RUN_ID=//p')"
PROJECTS+=("$(project_of "$RUN_ID")")
echo "    RUN_ID=$RUN_ID"

echo "==> fork A: rewind to fix_committed, continue with mock path 2 (fix B)"
BRANCH_A="$(uv run python -m rewind.fork --run "$RUN_ID" --checkpoint fix_committed --path path2 \
  | sed -n 's/^BRANCH_ID=//p')"
PROJECTS+=("$(project_of "$BRANCH_A")")

echo "==> fork B: rewind to pr_open (before CI), continue with the rest of path 1"
BRANCH_B="$(uv run python -m rewind.fork --run "$RUN_ID" --checkpoint pr_open --path path1 \
  | sed -n 's/^BRANCH_ID=//p')"
PROJECTS+=("$(project_of "$BRANCH_B")")

echo "==> checking results"
set +e
uv run python "$SCENARIO/check.py" --run "$RUN_ID" --branch-a "$BRANCH_A" --branch-b "$BRANCH_B"
CHECK_RC=$?
set -e

echo
if [ "$CHECK_RC" -eq 0 ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; fi
exit "$CHECK_RC"
