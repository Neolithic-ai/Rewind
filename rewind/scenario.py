"""Load a scenario directory (`scenario.yaml` + the files it references).

A scenario is data: which compose file to run, what to capture per service,
pre-snapshot consistency hooks, readiness checks, the seed steps, milestone
detectors (plus an optional shared `detect_prelude`), the scorer, the agent prompt
and optional mock-model paths. Scenario files belong to the calling experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Command:
    """Bash to run in a service: inline (`run`) or a scenario file (`script`)."""

    service: str
    run: str | None = None
    script: str | None = None
    user: str | None = None

    def source(self, root: Path) -> str:
        return (root / self.script).read_text() if self.script else (self.run or "")


@dataclass(frozen=True)
class Service:
    name: str
    capture: list[str]
    ready: str | None = None
    pre_snapshot: Command | None = None


@dataclass(frozen=True)
class SeedStep:
    name: str
    service: str
    run: str | None = None
    script: str | None = None
    user: str | None = None
    copy: str | None = None  # host path (relative to the scenario dir)
    to: str | None = None    # destination inside the container
    save_as: str | None = None


@dataclass(frozen=True)
class Milestone:
    name: str
    service: str
    detect: str


@dataclass(frozen=True)
class MockPath:
    name: str
    steps: list[str]
    final: str
    description: str = ""
    start_from: str | None = None  # milestone label for continuation scripts


@dataclass
class Scenario:
    root: Path
    name: str
    compose_file: Path
    build: list[str]
    vars: dict[str, str]
    services: dict[str, Service]
    seed: list[SeedStep]
    milestones: list[Milestone]
    scorer: Command
    agent: dict
    mock_paths: dict[str, MockPath] = field(default_factory=dict)
    fingerprint: Command | None = None
    # fingerprint keys that background services may still change on their own
    # (left out when a restored or finished environment is compared)
    fingerprint_volatile: list[str] = field(default_factory=list)
    detect_prelude: str = ""  # bash prepended to every milestone detector

    @property
    def prompt(self) -> str:
        return self.agent["prompt"].strip()

    @property
    def system(self) -> str | None:
        s = self.agent.get("system")
        return s.strip() if s else None


def _command(d: dict | None, default_service: str | None = None) -> Command | None:
    if d is None:
        return None
    return Command(service=d.get("service", default_service), run=d.get("run"),
                   script=d.get("script"), user=d.get("user"))


def load_scenario(path: str | Path) -> Scenario:
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    spec = yaml.safe_load((root / "scenario.yaml").read_text())

    services = {
        name: Service(name=name, capture=list(s.get("capture", [])), ready=s.get("ready"),
                      pre_snapshot=_command(s.get("pre_snapshot"), name))
        for name, s in spec["services"].items()
    }
    mock_paths: dict[str, MockPath] = {}
    if spec.get("mock_paths"):
        raw = yaml.safe_load((root / spec["mock_paths"]).read_text())
        for name, p in raw.items():
            mock_paths[name] = MockPath(name=name, steps=list(p["steps"]),
                                        final=p.get("final", "Done."),
                                        description=p.get("description", ""),
                                        start_from=p.get("from"))

    return Scenario(
        root=root,
        name=spec["name"],
        compose_file=root / spec.get("compose", "compose.yaml"),
        build=list(spec.get("build", [])),
        vars={k: str(v) for k, v in (spec.get("vars") or {}).items()},
        services=services,
        seed=[SeedStep(**s) for s in spec.get("seed", [])],
        milestones=[Milestone(**m) for m in spec.get("milestones", [])],
        scorer=_command(spec["scorer"]),
        agent=spec["agent"],
        mock_paths=mock_paths,
        fingerprint=_command(spec.get("fingerprint")),
        fingerprint_volatile=list((spec.get("fingerprint") or {}).get("volatile", [])),
        detect_prelude=spec.get("detect_prelude") or "",
    )
