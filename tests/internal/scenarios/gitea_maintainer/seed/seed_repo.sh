# Seed (runs on the workstation as the admin): org acme, repo acme/calc with the
# buggy package, the agent as a collaborator, issue #1 and the CI webhook.
set -euo pipefail
api() {
  curl -sSf -H "Authorization: token $ADMIN_TOKEN" -H 'Content-Type: application/json' "$@"
}
A="$GITEA_URL/api/v1"
api -X POST "$A/orgs" -d '{"username": "acme", "visibility": "public"}' >/dev/null
api -X POST "$A/orgs/acme/repos" \
  -d '{"name": "calc", "description": "Tiny statistics helpers", "private": false, "default_branch": "main"}' >/dev/null
api -X PUT "$A/repos/acme/calc/collaborators/$AGENT_USER" -d '{"permission": "write"}'
# The webhook exists before the first push, so CI's baseline run on main (red:
# the bug is already there) is part of the seeded state.
api -X POST "$A/repos/acme/calc/hooks" -d @- >/dev/null <<'JSON'
{"type": "gitea", "active": true, "branch_filter": "*", "events": ["push", "pull_request"],
 "config": {"url": "http://ci:8080/hook", "content_type": "json"}}
JSON

chown -R root:root /tmp/seed-calc && cd /tmp/seed-calc
git init -q -b main
git add -A
git -c user.name="Acme Dev" -c user.email=dev@acme.test commit -qm "calc 0.1.0"
git push -q "http://$ADMIN_USER:$ADMIN_TOKEN@${GITEA_URL#http://}/acme/calc.git" main
cd / && rm -rf /tmp/seed-calc

api -X POST "$A/repos/acme/calc/issues" -d @- >/dev/null <<'JSON'
{"title": "mean() returns wrong values",
 "body": "`calc.mean([2, 4, 6])` returns `6.0` instead of `4.0`, and `variance()` is wrong as a result.\n\n`python3 -m pytest -q` shows 2 failing tests (`test_mean`, `test_variance`)."}
JSON
# Wait for that baseline CI run so `seeded` is a quiet, deterministic state.
sha=$(git ls-remote "$GITEA_URL/acme/calc.git" refs/heads/main | cut -f1)
for i in $(seq 60); do
  # (Gitea flags the repo non-empty asynchronously; until then this can 500.)
  state=$(api "$A/repos/acme/calc/commits/$sha/status" 2>/dev/null | jq -r .state || true)
  [ "$state" = failure ] && break; sleep 1
done
[ "$state" = failure ] || { echo "baseline CI run did not finish (state=$state)"; exit 1; }
echo seeded
