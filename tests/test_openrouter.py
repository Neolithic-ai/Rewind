"""OpenRouter attribution headers reach the wire (no real key, no network).

Points Inspect's OpenRouter provider at a local HTTP stub that records request
headers and answers with a minimal chat completion, then generates through
`rewind.openrouter.resolve_model` (what `rewind.run`/`rewind.fork` use). Also
checks that the cost OpenRouter reports per call is read back from Inspect's log
(what the `--budget-usd` guard relies on).

    uv run --extra openrouter python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.solver import solver

from rewind import openrouter

COMPLETION = {
    "id": "gen-1", "object": "chat.completion", "created": 0, "model": "test/model",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "pong"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4, "cost": 0.0125},
}


class Stub(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict[str, str]]] = []

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        Stub.requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
        body = json.dumps(COMPLETION).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


class AttributionHeaders(unittest.TestCase):
    def setUp(self) -> None:
        Stub.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/api/v1"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def generate(self, model: str, env: dict[str, str]) -> dict[str, str]:
        with mock.patch.dict(os.environ, env, clear=False):
            for var in ("OPENROUTER_APP_URL", "OPENROUTER_APP_TITLE"):
                if var not in env:
                    os.environ.pop(var, None)
            m = openrouter.resolve_model(model, base_url=self.base_url)
            out = asyncio.run(m.generate("ping"))
        self.assertEqual(out.completion, "pong")
        self.assertEqual(len(Stub.requests), 1)
        path, headers = Stub.requests[0]
        self.assertEqual(path, "/api/v1/chat/completions")
        self.assertEqual(headers["authorization"], "Bearer dummy-key")
        return headers

    def test_default_headers(self) -> None:
        h = self.generate("openrouter/test/model-a", {"OPENROUTER_API_KEY": "dummy-key"})
        self.assertEqual(h["http-referer"], "https://github.com/Neolithic-ai/Rewind")
        self.assertEqual(h["x-openrouter-title"], "Rewind")
        self.assertEqual(h["x-title"], "Rewind")

    def test_headers_from_env(self) -> None:
        h = self.generate("openrouter/test/model-b", {
            "OPENROUTER_API_KEY": "dummy-key",
            "OPENROUTER_APP_URL": "https://github.com/acme/fork",
            "OPENROUTER_APP_TITLE": "fork"})
        self.assertEqual(h["http-referer"], "https://github.com/acme/fork")
        self.assertEqual(h["x-openrouter-title"], "fork")
        self.assertEqual(h["x-title"], "fork")

    def test_cost_from_logged_calls(self) -> None:
        @solver
        def two_calls():
            async def solve(state, generate):
                for _ in range(2):
                    await get_model().generate("ping")
                return state
            return solve

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "dummy-key"}):
            m = openrouter.resolve_model("openrouter/test/model-d", base_url=self.base_url)
            with tempfile.TemporaryDirectory() as d:
                log = inspect_eval(Task(dataset=[Sample(input="x")], solver=two_calls()),
                                   model=m, log_dir=d, display="none")[0]
                self.assertEqual(log.status, "success")
                cost = openrouter.events_cost(log.samples[0].events)  # loaded from d
        self.assertEqual(len(Stub.requests), 2)
        title = openrouter.attribution_headers()["X-Title"]
        self.assertTrue(all(h["x-title"] == title for _, h in Stub.requests))
        self.assertAlmostEqual(cost, 0.025)
        self.assertIsNone(openrouter.events_cost([]))

    def test_missing_key_fails_clearly(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENROUTER_API_KEY", None)
            with self.assertRaises(SystemExit) as cm:
                openrouter.resolve_model("openrouter/test/model-c", base_url=self.base_url)
        self.assertIn("OPENROUTER_API_KEY is not set", str(cm.exception))
        self.assertEqual(Stub.requests, [])

    def test_other_models_unchanged(self) -> None:
        self.assertEqual(openrouter.model_args("mockllm/model"), {})
        self.assertEqual(openrouter.model_args("anthropic/claude-sonnet-5"), {})


if __name__ == "__main__":
    unittest.main()
