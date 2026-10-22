"""A tiny CI service for the gitea_maintainer scenario (stdlib only).

Gitea webhook -> queue a job -> clone the commit -> `python3 -m pytest` -> post a
commit status (context `ci/pytest`) back through the Gitea API.

Triggers: `pull_request` (opened / synchronized / reopened) on the PR head, and
`push` to `refs/heads/main`. Other pushes (feature branches, tags) are ignored.

All state lives under CI_STATE_DIR (a named volume, so it is part of every
snapshot):

    config.json      {"token": ...}  written by the scenario seed
    next_id          job counter
    queue/<id>.json  pending jobs (removed when finished)
    runs/<id>.json   run history: id, kind, sha, ref, state, timestamps
    logs/<id>.log    pytest output

The queue is on disk, so a job that was queued when a snapshot was taken is
picked up again when the snapshot is restored (processes are not restored).
Jobs are de-duplicated on (kind, sha), so a webhook Gitea redelivers after a
restore does not run twice.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GITEA = os.environ.get("CI_GITEA_URL", "http://gitea:3000")
REPO = os.environ.get("CI_REPO", "acme/calc")
STATE = Path(os.environ.get("CI_STATE_DIR", "/var/lib/ci"))
DELAY = float(os.environ.get("CI_QUEUE_DELAY", "0"))
CONTEXT = "ci/pytest"

LOCK = threading.Lock()
WAKE = threading.Event()


def log(msg: str) -> None:
    print(f"[ci] {msg}", flush=True)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)  # atomic, so a snapshot never sees a half-written file


def token() -> str:
    try:
        return json.loads((STATE / "config.json").read_text())["token"]
    except Exception:
        return ""


def enqueue(kind: str, sha: str, ref: str) -> int | None:
    with LOCK:
        for f in (STATE / "runs").glob("*.json"):
            r = json.loads(f.read_text())
            if r["kind"] == kind and r["sha"] == sha:
                return None  # duplicate delivery
        counter = STATE / "next_id"
        job_id = int(counter.read_text()) if counter.exists() else 1
        counter.write_text(str(job_id + 1))
        run = {"id": job_id, "kind": kind, "sha": sha, "ref": ref,
               "state": "queued", "queued_at": time.time()}
        write_json(STATE / "runs" / f"{job_id}.json", run)
        write_json(STATE / "queue" / f"{job_id}.json", run)
    log(f"queued #{job_id} {kind} {sha[:10]}")
    WAKE.set()
    return job_id


def post_status(sha: str, state: str, description: str) -> None:
    body = json.dumps({"state": state, "context": CONTEXT,
                       "description": description}).encode()
    for attempt in range(60):  # Gitea may still be booting after a restore
        try:
            req = urllib.request.Request(
                f"{GITEA}/api/v1/repos/{REPO}/statuses/{sha}", data=body, method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"token {token()}"})
            urllib.request.urlopen(req, timeout=10).read()
            return
        except Exception as e:
            log(f"post status failed ({e}); retrying")
            time.sleep(1)


def run_job(job: dict) -> None:
    job_id, sha = job["id"], job["sha"]
    time.sleep(DELAY)
    run_file = STATE / "runs" / f"{job_id}.json"
    run = json.loads(run_file.read_text())
    run.update(state="running", started_at=time.time())
    write_json(run_file, run)
    post_status(sha, "pending", f"run #{job_id} started")

    work = Path(f"/tmp/ci-work/{job_id}")
    shutil.rmtree(work, ignore_errors=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    checkout = f"git clone -q {GITEA}/{REPO}.git {work} && git -C {work} checkout -q {sha}"
    for attempt in range(30):  # Gitea may still be booting after a restore
        if subprocess.run(["bash", "-c", checkout], capture_output=True).returncode == 0:
            break
        shutil.rmtree(work, ignore_errors=True)
        time.sleep(1)
    proc = subprocess.run(["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                          cwd=work, capture_output=True, text=True)
    (STATE / "logs" / f"{job_id}.log").write_text(proc.stdout + proc.stderr)
    state = "success" if proc.returncode == 0 else "failure"
    summary = (proc.stdout.strip().splitlines() or ["no output"])[-1]
    post_status(sha, state, f"run #{job_id}: {summary}"[:200])

    run.update(state=state, finished_at=time.time(), summary=summary)
    write_json(run_file, run)
    (STATE / "queue" / f"{job_id}.json").unlink(missing_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    log(f"finished #{job_id} {state}: {summary}")


def worker() -> None:
    while True:
        pending = sorted((STATE / "queue").glob("*.json"), key=lambda p: int(p.stem))
        if not pending:
            WAKE.wait(timeout=2)
            WAKE.clear()
            continue
        try:
            run_job(json.loads(pending[0].read_text()))
        except Exception as e:  # keep the worker alive; leave the job queued
            log(f"job {pending[0].name} crashed: {e}")
            time.sleep(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _reply(self, code: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._reply(200, {"ok": True})
        if self.path == "/runs":
            runs = [json.loads(f.read_text()) for f in (STATE / "runs").glob("*.json")]
            return self._reply(200, sorted(runs, key=lambda r: r["id"]))
        self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/hook":
            return self._reply(404, {"error": "not found"})
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        event = self.headers.get("X-Gitea-Event", "")
        job = None
        if event == "pull_request" and payload.get("action") in ("opened", "synchronized", "reopened"):
            pr = payload["pull_request"]
            job = enqueue("pull_request", pr["head"]["sha"], pr["head"]["ref"])
        elif event == "push" and payload.get("ref") == "refs/heads/main" and payload.get("after"):
            job = enqueue("push", payload["after"], "main")
        self._reply(200, {"queued": job})


def main() -> None:
    for sub in ("queue", "runs", "logs"):
        (STATE / sub).mkdir(parents=True, exist_ok=True)
    resumed = len(list((STATE / "queue").glob("*.json")))
    if resumed:
        log(f"resuming {resumed} queued job(s) from disk")
    threading.Thread(target=worker, daemon=True).start()
    log("listening on :8080")
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()


if __name__ == "__main__":
    main()
