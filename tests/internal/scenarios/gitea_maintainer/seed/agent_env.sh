# Give the agent its credentials. /root/.agent_env is the workstation's BASH_ENV
# (see compose.yaml), so every `bash -c` the agent runs sources it.
set -euo pipefail
cat > /root/.agent_env <<ENV
export GITEA_URL=$GITEA_URL
export GITEA_USER=$AGENT_USER
export GITEA_TOKEN=$AGENT_TOKEN
gapi() {
  local method=\$1 path=\$2; shift 2
  curl -sS -X "\$method" -H "Authorization: token \$GITEA_TOKEN" \\
    -H 'Content-Type: application/json' "\$GITEA_URL/api/v1\$path" "\$@"
}
ENV
git config --global user.name "$AGENT_USER"
git config --global user.email agent@acme.test
git config --global init.defaultBranch main
git config --global credential.helper store
echo "http://$AGENT_USER:$AGENT_TOKEN@${GITEA_URL#http://}" > /root/.git-credentials
chmod 600 /root/.git-credentials
echo ok
