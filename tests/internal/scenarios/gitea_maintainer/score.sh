# Scorer: verify the end state via the Gitea API (runs on the workstation with
# the admin token). Prints one JSON object; `success` is the overall verdict.
set -u
API="$GITEA_URL/api/v1/repos/acme/calc"
get() { curl -sf -H "Authorization: token $ADMIN_TOKEN" "$API$1"; }

pr=$(get "/pulls?state=closed&limit=50" \
  | jq -c '[.[] | select(.merged and ((.title + " " + (.body // "")) | test("#1\\b")))] | first // {}')
pr_number=$(jq -r '.number // empty' <<<"$pr")
merge_sha=$(jq -r '.merge_commit_sha // empty' <<<"$pr")
merge_ci=$( [ -n "$merge_sha" ] && get "/commits/$merge_sha/status" | jq -r '.state' || echo none)
issue_state=$(get "/issues/1" | jq -r '.state')
tag_sha=$(get "/tags/v0.1.1" | jq -r '.commit.sha // empty')

tests_at_tag=false
if [ -n "$tag_sha" ]; then
  tmp=$(mktemp -d)
  if git clone -q --depth 1 --branch v0.1.1 "$GITEA_URL/acme/calc.git" "$tmp/calc" 2>/dev/null \
     && (cd "$tmp/calc" && python3 -m pytest -q -p no:cacheprovider >/dev/null 2>&1); then
    tests_at_tag=true
  fi
  rm -rf "$tmp"
fi

jq -n --arg pr "$pr_number" --arg sha "$merge_sha" --arg ci "$merge_ci" \
      --arg issue "$issue_state" --arg tag "$tag_sha" --argjson tests "$tests_at_tag" '{
  pr_merged: ($pr != ""), pr_number: $pr, merge_commit_sha: $sha,
  merge_ci_success: ($ci == "success"), merge_ci_state: $ci,
  issue_closed: ($issue == "closed"), tag_exists: ($tag != ""), tag_sha: $tag,
  tests_pass_at_tag: $tests
} | .success = (.pr_merged and .merge_ci_success and .issue_closed and .tag_exists and .tests_pass_at_tag)'
