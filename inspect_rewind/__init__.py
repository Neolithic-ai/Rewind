"""inspect-rewind: rewind and branch any Inspect task that uses a docker sandbox.

    inspect eval task.py --sandbox rewind ...           # snapshot after every tool call
    inspect-rewind fork <log.eval> --sample ID --step N  # branch from step N

See README.md for model switching and continuing branches.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .fork import fork, mock_outputs, rewind_seed, snapshots
    from .sandbox import RewindSandboxEnvironment

__all__ = ["RewindSandboxEnvironment", "fork", "mock_outputs", "rewind_seed", "snapshots"]


def _load_exports() -> None:
    try:
        branching = import_module(".fork", __name__)
        sandbox = import_module(".sandbox", __name__)
    except ModuleNotFoundError as exc:
        if exc.name == "inspect_ai":
            raise ModuleNotFoundError(
                "The Inspect adapter requires the optional Inspect dependency. "
                "Install it with: pip install 'rewind-env[inspect]' "
                "(or pip install '.[inspect]' from a checkout).",
                name="inspect_ai",
            ) from exc
        raise

    # Set the complete public API together: importing the fork submodule also
    # creates a package attribute named 'fork', which must expose the function.
    for export in __all__:
        globals()[export] = getattr(sandbox if export == "RewindSandboxEnvironment"
                                   else branching, export)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _load_exports()
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


try:
    _load_exports()
except ImportError:
    # Help and doctor must remain available with missing or incompatible
    # optional dependencies. Accessing adapter APIs retries and reports the
    # actual import failure. The core rewind package never imports this one.
    pass
