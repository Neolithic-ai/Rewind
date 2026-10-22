#!/usr/bin/env bash
# End-to-end PoC: run the scripted agent, checkpoint every step, rewind to step 3
# and branch, then verify. Exits non-zero on any unexpected FAIL.
#
#   bash tests/internal/poc/run_all.sh
#
# Env:
#   KEEP=1       leave the Docker environments running after the run
#   SKIP_PROBE=1 skip the (best-effort) Inspect-native-checkpointing probe
set -euo pipefail

cd "$(dirname "$0")/../../.."
export PATH="$HOME/.local/bin:$PATH"

echo "==> syncing deps (uv)"
uv sync --quiet

echo "==> building sandbox images"
docker compose -p rewind_build -f tests/internal/env/compose.yaml build default >/dev/null
docker compose -p rewind_build -f tests/internal/env/compose.yaml down -v -t2 >/dev/null 2>&1 || true

echo "==> running original task (6 steps, checkpoint after each)"
RUN_ID="$(uv run python -m tests.internal.poc.task | sed -n 's/^RUN_ID=//p')"
echo "    RUN_ID=$RUN_ID"

echo "==> rewinding to step 3 and branching (alt script)"
BRANCH_ID="$(uv run python -m tests.internal.poc.fork --run "$RUN_ID" --step 3 --script alt | sed -n 's/^BRANCH_ID=//p')"
echo "    BRANCH_ID=$BRANCH_ID"

echo "==> checking results"
set +e
uv run python -m tests.internal.poc.check --run "$RUN_ID" --branch "$BRANCH_ID"
CHECK_RC=$?
set -e

if [ "${SKIP_PROBE:-0}" != "1" ]; then
  echo
  echo "==> (best-effort) Inspect native-checkpointing probe"
  uv run python -m tests.internal.poc.inspect_probe || true
fi

if [ "${KEEP:-0}" != "1" ]; then
  echo
  echo "==> tearing down environments"
  for proj in "rewind_${RUN_ID}" "rewind_${BRANCH_ID}"; do
    docker compose -p "$proj" -f tests/internal/env/compose.yaml down -v -t2 >/dev/null 2>&1 || true
  done
fi

echo
if [ "$CHECK_RC" -eq 0 ]; then
  echo "RESULT: PASS"
else
  echo "RESULT: FAIL ($CHECK_RC unexpected result(s))"
fi
exit "$CHECK_RC"
