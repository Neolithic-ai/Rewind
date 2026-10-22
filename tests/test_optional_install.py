"""The environment engine works without installing the Inspect adapter."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NO_INSPECT = """
import importlib.abc
import sys

class NoInspect(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "inspect_ai" or fullname.startswith("inspect_ai."):
            raise ModuleNotFoundError("Inspect is deliberately absent", name="inspect_ai")

sys.meta_path.insert(0, NoInspect())
"""


class OptionalAdapter(unittest.TestCase):
    def run_python(self, code: str, *, without_inspect: bool = False) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [sys.executable, "-c", (NO_INSPECT if without_inspect else "") + textwrap.dedent(code)],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_core_and_adapter_namespace_import_without_inspect(self) -> None:
        self.run_python("""
            import rewind
            import rewind.env
            import inspect_rewind
            from rewind import Checkpoint, Environment
            assert callable(Checkpoint) and callable(Environment)
            assert "inspect_ai" not in sys.modules
            assert set(inspect_rewind.__all__) == {
                "RewindSandboxEnvironment", "fork", "mock_outputs", "rewind_seed", "snapshots"
            }
        """, without_inspect=True)

    def test_missing_adapter_dependency_has_install_guidance(self) -> None:
        self.run_python("""
            import inspect_rewind
            try:
                inspect_rewind.snapshots
            except ModuleNotFoundError as exc:
                assert exc.name == "inspect_ai"
                assert "rewind-env[inspect]" in str(exc)
            else:
                raise AssertionError("Expected the missing optional dependency")
        """, without_inspect=True)

    def test_existing_public_exports_remain_callable(self) -> None:
        for first in ("fork", "snapshots", "RewindSandboxEnvironment"):
            with self.subTest(first=first):
                self.run_python(f"""
                    import inspect_rewind
                    getattr(inspect_rewind, {first!r})
                    from inspect_rewind import fork, mock_outputs, rewind_seed, snapshots
                    from inspect_rewind import RewindSandboxEnvironment
                    from inspect_rewind.fork import prefix_for_step
                    assert all(callable(value) for value in (
                        fork, mock_outputs, rewind_seed, snapshots, RewindSandboxEnvironment
                    ))
                    assert inspect_rewind.fork is fork
                """)

    def test_cli_help_does_not_require_inspect(self) -> None:
        for module in ("rewind.cli", "inspect_rewind.cli"):
            with self.subTest(module=module):
                result = self.run_python(f"""
                    import runpy
                    sys.argv = [{module!r}, "--help"]
                    runpy.run_module({module!r}, run_name="__main__")
                """, without_inspect=True)
                self.assertIn("usage:", result.stdout)

    def test_explicit_submodule_import_keeps_public_fork_callable(self) -> None:
        self.run_python("""
            from inspect_rewind.cli import checkpoint_report
            from inspect_rewind.fork import prefix_for_step
            from inspect_rewind import fork
            from importlib import import_module
            assert callable(fork)
            assert fork is import_module("inspect_rewind.fork").fork
        """)

    def test_metadata_keeps_the_adapter_optional_and_development_ready(self) -> None:
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        project = config["project"]
        self.assertFalse(any(dep.startswith(("inspect-ai", "openai"))
                             for dep in project["dependencies"]))
        extra = project["optional-dependencies"]["inspect"]
        self.assertEqual(project["name"], "rewind-env")
        self.assertEqual(len(extra), 1)
        self.assertRegex(extra[0], r"^inspect-ai==[0-9]+\.[0-9]+\.[0-9]+$")
        self.assertIn(extra[0], project["optional-dependencies"]["openrouter"])
        self.assertIn(extra[0], config["dependency-groups"]["dev"])
        self.assertIn("openai>=3.4.0", config["dependency-groups"]["dev"])
        self.assertEqual(project["scripts"]["rewind"], "rewind.cli:main")
        self.assertEqual(project["scripts"]["inspect-rewind"], "inspect_rewind.cli:main")


if __name__ == "__main__":
    unittest.main()
