"""Validate resolved Compose configuration before provisioning a branch."""

from __future__ import annotations

from typing import Any


def validate_compose_isolation(config: dict[str, Any], project: str) -> None:
    """Require storage and namespaces to belong to this Compose project.

    ``config`` must be Docker Compose's resolved JSON, including generated
    resource names. Read-only bind mounts are allowed for immutable inputs;
    writable state must use the container filesystem or private local volumes.
    This is branch isolation validation, not a container security boundary.
    """
    for service, spec in (config.get("services") or {}).items():
        if spec.get("container_name"):
            raise ValueError(f"service {service!r}: remove container_name so branches can use unique containers")
        for namespace in ("network_mode", "pid", "ipc"):
            mode = spec.get(namespace) or ""
            shared = mode == "host" or mode.startswith("container:")
            if namespace == "network_mode" and mode:
                # Named networks here (including Docker's global "bridge")
                # bypass Compose's project-scoped networks entirely.
                shared = mode != "none" and not mode.startswith("service:")
            if shared:
                raise ValueError(f"service {service!r}: {namespace}={mode!r} shares a namespace "
                                 "outside this project; use a private or project service namespace")
        if any(source.startswith("container:") for source in spec.get("volumes_from") or []):
            raise ValueError(f"service {service!r}: volumes_from another container shares writable "
                             "storage across branches; use project-managed volumes")
        for mount in spec.get("volumes") or []:
            if mount.get("type") == "bind" and not mount.get("read_only"):
                raise ValueError(f"service {service!r}: writable bind mount {mount.get('target')!r} "
                                 "is shared across branches; use a named volume or a read-only bind")
    for kind in ("volumes", "networks"):
        for name, spec in (config.get(kind) or {}).items():
            spec = spec or {}
            if spec.get("external") or spec.get("name", f"{project}_{name}") != f"{project}_{name}":
                raise ValueError(f"{kind} {name!r} is shared across projects; remove external/name "
                                 "so Compose creates a separate resource for each branch")
            if kind == "volumes" and (spec.get("driver_opts") or spec.get("driver", "local") != "local"):
                raise ValueError(f"volume {name!r} uses custom backing storage that may be shared "
                                 "across branches; use a local volume without driver_opts")
