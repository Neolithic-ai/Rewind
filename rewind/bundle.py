"""Portable, verified checkpoint archives without an agent-framework dependency."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tarfile
import tempfile
from typing import Any, Iterator

import yaml

from .env import _run


def _relative(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or PureWindowsPath(name).drive
            or '..' in path.parts or '\\' in name
            or str(path) != name or name == '.'):
        raise ValueError(f"unsafe bundle path: {name!r}")
    return name


def _digest(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def copy_asset(source: Path, destination: Path) -> None:
    """Copy explicit inputs only; links and special files are not portable assets."""
    if source.is_symlink():
        raise ValueError(f"bundle input is a symlink: {source}; use a regular file or directory")
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=False)
        for item in sorted(source.iterdir()):
            if item.name in {'.git', '.env', '.venv', '__pycache__'} or item.name.startswith('.env.'):
                raise ValueError(f"bundle input includes {item.name}; select a narrower input directory")
            copy_asset(item, destination / item.name)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
    else:
        raise ValueError(f"bundle input must be a regular file or directory: {source}")


def write_bundle(staging: Path, output: Path, manifest: dict[str, Any]) -> Path:
    """Publish a new archive with a complete SHA-256 inventory; never overwrite."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"bundle destination already exists: {output}")
    files = {}
    directories = []
    for path in sorted(staging.rglob('*')):
        if path.is_symlink():
            raise ValueError(f"bundle cannot contain symlinks: {path}")
        if path.is_dir():
            directories.append(_relative(path.relative_to(staging).as_posix()))
            continue
        if not path.is_file():
            raise ValueError(f"bundle cannot contain special files: {path}")
        name = _relative(path.relative_to(staging).as_posix())
        if name == 'bundle.json':
            raise ValueError('bundle.json is reserved for the bundle manifest')
        files[name] = {'sha256': _digest(path), 'size': path.stat().st_size}
    manifest = {**manifest, 'schema_version': 1, 'files': files, 'directories': directories}
    (staging / 'bundle.json').write_text(json.dumps(manifest, indent=2) + '\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rewind-export-', dir=output.parent) as tmp:
        archive_path = Path(tmp) / 'bundle'
        with tarfile.open(archive_path, 'w:gz') as archive:
            for name in ['bundle.json', *files]:
                archive.add(staging / name, arcname=name, recursive=False)
        os.link(archive_path, output)  # atomic, and fails if another writer won
    return output


@contextmanager
def extracted_bundle(archive: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Validate and extract regular files to private staging; never execute inputs."""
    with tempfile.TemporaryDirectory(prefix='rewind-import-') as tmp:
        staging = Path(tmp)
        try:
            with tarfile.open(archive, 'r:*') as source:
                members = source.getmembers()
                names: set[str] = set()
                for item in members:
                    _relative(item.name)
                    if not item.isfile() or item.name in names:
                        raise ValueError(f"bundle has duplicate or non-regular entry: {item.name}")
                    names.add(item.name)
                manifests = [m for m in members if m.name == 'bundle.json']
                if len(manifests) != 1 or manifests[0].size > 8 * 1024 * 1024:
                    raise ValueError('bundle needs one reasonably sized bundle.json manifest')
                manifest = json.load(source.extractfile(manifests[0]))
                if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
                    raise ValueError('unsupported bundle schema')
                files = manifest.get('files')
                if not isinstance(files, dict) or set(files) != names - {'bundle.json'}:
                    raise ValueError('bundle contents do not match the manifest inventory')
                directories = manifest.get('directories', [])
                if (not isinstance(directories, list)
                        or not all(isinstance(name, str) for name in directories)):
                    raise ValueError('invalid bundle directory inventory')
                for name in directories:
                    _relative(name)
                    if name in names:
                        raise ValueError(f"bundle directory conflicts with a file: {name}")
                    (staging / name).mkdir(parents=True, exist_ok=True)
                for item in members:
                    if item.name == 'bundle.json':
                        continue
                    expected = files[item.name]
                    if (not isinstance(expected, dict) or expected.get('size') != item.size
                            or not isinstance(expected.get('sha256'), str)):
                        raise ValueError(f"invalid inventory entry for {item.name}")
                    target = staging / item.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as out, source.extractfile(item) as inp:
                        shutil.copyfileobj(inp, out)
                    if _digest(target) != expected['sha256']:
                        raise ValueError(f"bundle hash verification failed: {item.name}")
                    target.chmod(0o755 if item.mode & 0o111 else 0o644)
                (staging / 'bundle.json').write_text(json.dumps(manifest, indent=2) + '\n')
        except (tarfile.TarError, EOFError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint bundle: {exc}") from exc
        yield staging, manifest


def publish_directory(staging: Path, output: Path) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"import destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rewind-import-', dir=output.parent) as tmp:
        pending = Path(tmp) / 'contents'
        shutil.copytree(staging, pending)
        if output.exists():
            raise FileExistsError(f"import destination already exists: {output}")
        pending.rename(output)
    return output


def stage_compose(compose_file: Path, destination: Path, *, project: str = 'rewind_export',
                  environment: dict[str, str] | None = None) -> Path:
    """Resolve Compose, copy declared read-only inputs, and pin existing images.

    Images are requirements, not embedded layers. A local-only image must be
    transferred separately with Docker save/load; no build context is bundled.
    """
    from .isolation import validate_compose_isolation

    compose_file = Path(compose_file).expanduser().resolve()
    result = _run(['docker', 'compose', '-p', project, '-f', str(compose_file),
                   'config', '--format', 'json'],
                  env={**os.environ, **environment} if environment is not None else None)
    config = json.loads(result.stdout)
    validate_compose_isolation(config, project)
    destination.mkdir(parents=True, exist_ok=True)
    config.pop('name', None)
    for kind in ('volumes', 'networks'):
        for spec in (config.get(kind) or {}).values():
            if spec:
                spec.pop('name', None)
    asset_count = 0

    def asset(value: str) -> str:
        nonlocal asset_count
        source = Path(value.replace('$$', '$')).expanduser()
        if not source.is_absolute():
            source = compose_file.parent / source
        asset_count += 1
        relative = f'assets/{asset_count:03d}/{source.name}'
        copy_asset(source, destination / relative)
        return './' + relative.replace('$', '$$')

    images = {}
    for service, spec in config['services'].items():
        image_ref = spec.get('image') or f'{project}-{service}'
        try:
            info = json.loads(_run(['docker', 'image', 'inspect', image_ref]).stdout)[0]
        except RuntimeError as exc:
            raise ValueError(f"image for service {service!r} is unavailable: {image_ref}; "
                             'pull/build it before exporting') from exc
        digests = info.get('RepoDigests') or []
        resolved = digests[0] if digests else info['Id']
        platform = '/'.join(filter(None, [info.get('Os'), info.get('Architecture'), info.get('Variant')]))
        if platform:
            requested = spec.get('platform')
            if requested and requested != platform and not platform.startswith(requested + '/'):
                raise ValueError(f"service {service!r}: local image platform {platform!r} differs "
                                 f"from requested {requested!r}; load the matching image before export")
            spec['platform'] = platform
        spec['image'] = resolved
        spec.pop('build', None)
        if not digests:
            spec['pull_policy'] = 'never'
        images[service] = {'original': image_ref, 'image': resolved,
                           'image_id': info['Id'], 'local_only': not bool(digests),
                           'platform': platform}
        for mount in spec.get('volumes') or []:
            if mount.get('type') == 'bind':
                mount['source'] = asset(mount['source'])
        # Compose resolves env_file into environment. Reject unexpected leftovers
        # instead of retaining inaccessible host paths.
        if spec.get('env_file'):
            raise ValueError(f"service {service!r}: Compose did not resolve env_file")
        if spec.get('label_file') or spec.get('extends') or spec.get('develop'):
            raise ValueError(f"service {service!r}: label_file/extends/develop must be resolved or removed before export")
        if spec.get('devices'):
            raise ValueError(f"service {service!r}: host devices cannot be bundled")
    for kind in ('configs', 'secrets'):
        for name, spec in (config.get(kind) or {}).items():
            if spec.get('external') or spec.get('environment'):
                raise ValueError(f"{kind} {name!r}: external/environment resources cannot be bundled")
            if spec.get('file'):
                spec['file'] = asset(spec['file'])
    path = destination / 'compose.yaml'
    # Compose's config serializer already escapes literal dollars for reuse.
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    (destination / 'images.json').write_text(json.dumps(images, indent=2) + '\n')
    return path


def export_checkpoint(checkpoint: Any, output: str | Path) -> Path:
    from .workspace import Checkpoint

    checkpoint = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint.load(checkpoint)
    checkpoint.verify()
    with tempfile.TemporaryDirectory(prefix='rewind-stage-') as tmp:
        staging = Path(tmp)
        stage_compose(Path(checkpoint.manifest['compose_file']), staging / 'environment',
                      project=checkpoint.manifest.get('project', 'rewind_export'))
        saved = staging / 'checkpoint'
        saved.mkdir()
        for name in checkpoint.manifest['content_sha256']:
            copy_asset(checkpoint.path / name, saved / name)
        metadata = {**checkpoint.manifest, 'compose_file': '../environment/compose.yaml'}
        (saved / 'meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return write_bundle(staging, Path(output), {'kind': 'checkpoint', 'checkpoint': 'checkpoint'})


def import_checkpoint(archive: str | Path, output: str | Path) -> Any:
    from .workspace import Checkpoint

    output = Path(output).expanduser().resolve()
    with extracted_bundle(Path(archive)) as (staging, manifest):
        if manifest.get('kind') != 'checkpoint' or manifest.get('checkpoint') != 'checkpoint':
            raise ValueError('expected a standalone checkpoint bundle; use inspect-rewind for Inspect bundles')
        meta_path = staging / 'checkpoint' / 'meta.json'
        metadata = json.loads(meta_path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError('checkpoint metadata must be a JSON object')
        metadata['compose_file'] = str(output / 'environment' / 'compose.yaml')
        meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
        Checkpoint.load(staging / 'checkpoint').verify()
        if not (staging / 'environment' / 'compose.yaml').is_file():
            raise ValueError('bundle is missing its environment configuration')
        publish_directory(staging, output)
    return Checkpoint.load(output / 'checkpoint')
