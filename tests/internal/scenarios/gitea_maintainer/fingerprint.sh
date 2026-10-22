# Visible state of acme/calc: PRs, tags, issue #1, webhooks and the CI run
# history in Gitea, plus every local clone on the workstation (wherever it is:
# HEAD, branch and a hash of its working tree, untracked files included). Runs on the workstation with the
# admin token. Read-only: git runs with --no-optional-locks.
set -u
API="$GITEA_URL/api/v1/repos/acme/calc"
get() { curl -sf -H "Authorization: token $ADMIN_TOKEN" "$API$1"; }
clones() {
  find / -xdev -maxdepth 6 \( -path /proc -o -path /sys -o -path /dev \) -prune \
      -o -name .git -type d -print 2>/dev/null | sort \
    | while read -r g; do
        d=${g%/.git}
        git -C "$d" remote get-url origin 2>/dev/null | grep -q 'acme/calc' || continue
        gq() { git --no-optional-locks -C "$d" "$@" 2>/dev/null; }
        tree=$( { gq status --porcelain -uall; gq diff HEAD --binary
                  gq ls-files -z -o | (cd "$d" && xargs -0 -r sha256sum); } | sha256sum | cut -c1-16)
        jq -n --arg path "$d" --arg head "$(gq rev-parse HEAD)" \
          --arg branch "$(gq symbolic-ref -q --short HEAD)" --arg tree "$tree" \
          '{path: $path, head: $head, branch: $branch, worktree: $tree}'
      done | jq -s .
}
jq -n \
  --argjson prs "$(get '/pulls?state=all&limit=50' | jq '[.[] | {number, title, state, merged,
      merge_commit_sha, head: .head.ref, head_sha: .head.sha}] | sort_by(.number)')" \
  --argjson tags "$(get '/tags' | jq '[.[] | {name, sha: .commit.sha}]')" \
  --argjson issue "$(get '/issues/1' | jq '{number, title, state}')" \
  --argjson hooks "$(get '/hooks' | jq '[.[] | {url: .config.url, events, active}]')" \
  --argjson main "$(get '/branches/main' | jq '.commit.id')" \
  --argjson ci "$(curl -sf http://ci:8080/runs | jq '[.[] | {id, kind, sha, ref, state}]')" \
  --argjson clones "$(clones)" \
  '{prs: $prs, tags: $tags, issue: $issue, hooks: $hooks, main: $main, ci_runs: $ci,
    clones: $clones}'
