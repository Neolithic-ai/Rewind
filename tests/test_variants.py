"""Branch-variant plumbing, without Docker: the seed transcript a variant branch
starts from (`branch_prefix`), the cross-provider adaptation of the parent's
messages (`adapt_for_model`), the transcript comparison that has to survive it
(`normalize`), reading back which model an Inspect log called
(`logged_models`), and the variants file (`rewind.evals.load_variants`).

    uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import solver
from inspect_ai.tool import ToolCall

from rewind.agent import MARK, adapt_for_model, branch_prefix, load_messages, logged_models, \
    normalize, save_messages
from rewind.evals import load_variants, variant_argv

REPO = Path(__file__).resolve().parents[1]


def parent_prefix() -> list:
    """An Anthropic-style transcript: thinking blocks and toolu_ ids."""
    return [
        ChatMessageSystem(content="You are a maintainer."),
        ChatMessageUser(content="Fix the issue."),
        ChatMessageAssistant(
            content=[ContentReasoning(reasoning="plan", signature="sig-abc"),
                     ContentText(text="Cloning.", internal={"provider": "x"})],
            tool_calls=[ToolCall(id="toolu_01AbC", function="bash", arguments={"cmd": "git clone"})],
            model="anthropic/claude-sonnet-5"),
        ChatMessageTool(content="ok\n[exit code 0]", tool_call_id="toolu_01AbC", function="bash"),
        ChatMessageAssistant(
            content=[ContentReasoning(reasoning="", redacted=True, signature="enc")],
            tool_calls=[ToolCall(id="toolu_02XyZ", function="bash", arguments={"cmd": "pytest"})],
            model="anthropic/claude-sonnet-5"),
        ChatMessageTool(content="2 failed\n[exit code 1]", tool_call_id="toolu_02XyZ",
                        function="bash"),
    ]


class BranchPrefix(unittest.TestCase):
    def test_inject_goes_right_after_the_prefix_and_is_marked(self) -> None:
        prefix = parent_prefix()
        seed, i = branch_prefix(prefix, inject="Write a regression test first.")
        self.assertEqual(i, len(prefix))
        self.assertEqual(seed[:i], prefix)
        self.assertEqual((seed[i].role, seed[i].text), ("user", "Write a regression test first."))
        self.assertEqual(seed[i].metadata, {MARK: "injected"})
        self.assertEqual(len(prefix), 6, "the parent's list is not modified")

    def test_system_override_replaces_or_adds_the_system_prompt(self) -> None:
        seed, i = branch_prefix(parent_prefix(), system="Be terse.")
        self.assertIsNone(i)
        self.assertEqual((seed[0].role, seed[0].text), ("system", "Be terse."))
        self.assertEqual(seed[0].metadata, {MARK: "system_override"})
        self.assertEqual(normalize(seed[1:]), normalize(parent_prefix()[1:]))

        no_system = parent_prefix()[1:]
        seed, i = branch_prefix(no_system, system="Be terse.", inject="x")
        self.assertEqual([m.role for m in seed[:2]], ["system", "user"])
        self.assertEqual(i, len(no_system) + 1)

    def test_no_options_is_the_plain_prefix(self) -> None:
        prefix = parent_prefix()
        self.assertEqual(branch_prefix(prefix), (prefix, None))


class AdaptForModel(unittest.TestCase):
    def test_reasoning_dropped_ids_renumbered_text_kept(self) -> None:
        out, stats = adapt_for_model(parent_prefix())
        self.assertEqual(stats, {"tool_call_ids_rewritten": 2, "reasoning_blocks_dropped": 2,
                                 "other_blocks_dropped": 0})
        self.assertEqual(out[2].content, "Cloning.")
        self.assertEqual(out[4].content, "")
        self.assertEqual([tc.id for tc in out[2].tool_calls + out[4].tool_calls],
                         ["call00001", "call00002"])
        self.assertEqual([out[3].tool_call_id, out[5].tool_call_id], ["call00001", "call00002"])
        self.assertTrue(all(len(tc.id) == 9 and tc.id.isalnum()
                            for m in out if m.role == "assistant" for tc in m.tool_calls))
        self.assertEqual(parent_prefix()[2].tool_calls[0].id, "toolu_01AbC")

    def test_adapted_prefix_still_compares_equal_to_the_parent(self) -> None:
        out, _ = adapt_for_model(parent_prefix())
        self.assertEqual(normalize(out), normalize(parent_prefix()))

    def test_adapted_prefix_round_trips_through_messages_json(self) -> None:
        out, _ = adapt_for_model(parent_prefix())
        with tempfile.TemporaryDirectory() as d:
            save_messages(out, Path(d) / "m.json")
            self.assertEqual(normalize(load_messages(Path(d) / "m.json")), normalize(out))

    def test_normalize_catches_a_tool_result_for_the_wrong_call(self) -> None:
        bad = parent_prefix()
        bad[3] = bad[3].model_copy(update={"tool_call_id": "toolu_02XyZ"})
        self.assertNotEqual(normalize(bad), normalize(parent_prefix()))


class LoggedModels(unittest.TestCase):
    def test_reads_the_called_model_from_the_inspect_log(self) -> None:
        @solver
        def one_call():
            async def solve(state, generate):
                await get_model().generate("ping")
                return state
            return solve

        model = get_model("mockllm/other-model",
                          custom_outputs=[ModelOutput.from_content("mockllm/other-model", "pong")])
        with tempfile.TemporaryDirectory() as d:
            log = inspect_eval(Task(dataset=[Sample(input="x")], solver=one_call()),
                               model=model, log_dir=d, display="none")[0]
            self.assertEqual(log.status, "success")
            self.assertEqual(logged_models(Path(d)), {"inspect": ["mockllm/other-model"],
                                                      "served": ["mockllm/other-model"]})
            self.assertEqual(logged_models(Path(d) / "missing"), {"inspect": [], "served": []})


class LoadVariants(unittest.TestCase):
    def write(self, text: str) -> Path:
        d = Path(tempfile.mkdtemp())
        (d / "v.yaml").write_text(text)
        return d / "v.yaml"

    def test_example_file(self) -> None:
        path = REPO / "tests/internal/scenarios/gitea_maintainer/variants.example.yaml"
        real = load_variants(path, "openrouter/anthropic/claude-sonnet-5")
        self.assertEqual([v["name"] for v in real], ["same", "alt-instruction", "other-model"])
        self.assertEqual([v["model"] for v in real],
                         ["openrouter/anthropic/claude-sonnet-5"] * 2
                         + ["openrouter/deepseek/deepseek-v3.2"])
        self.assertIn("regression test", real[1]["inject"])
        self.assertEqual(variant_argv(real[0]), ["--variant", "same"])
        self.assertEqual(variant_argv(real[1])[2], "--inject")

        mock = load_variants(path, "mockllm/model")
        self.assertEqual([v["model"] for v in mock],
                         ["mockllm/model", "mockllm/model", "mockllm/deepseek-v3.2"])
        self.assertEqual(mock[2]["model_requested"], "openrouter/deepseek/deepseek-v3.2")

    def test_no_file_is_one_plain_variant(self) -> None:
        self.assertEqual(load_variants(None, "m"), [{"name": None, "model": "m"}])
        self.assertEqual(variant_argv({"name": None, "model": "m"}), [])

    def test_inject_file_and_system(self) -> None:
        path = self.write("- {name: a, inject_file: msg.txt, system: Be terse.}\n")
        (path.parent / "msg.txt").write_text("from a file")
        [v] = load_variants(path, "mockllm/model")
        self.assertEqual(v["inject"], "from a file")
        self.assertEqual(variant_argv(v), ["--variant", "a", "--inject-file",
                                           str(path.parent / "msg.txt"), "--system", "Be terse."])

    def test_bad_files_fail_clearly(self) -> None:
        for text in ("[]\n", "- {model: x}\n", "- {name: a, colour: red}\n",
                     "- {name: a}\n- {name: a}\n", "- {name: a, inject: x, inject_file: y}\n"):
            with self.assertRaises(SystemExit, msg=text):
                load_variants(self.write(text), "mockllm/model")


if __name__ == "__main__":
    unittest.main()
