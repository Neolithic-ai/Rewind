"""Archive validation and real export/import with the original files removed."""
from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import yaml

from rewind import Checkpoint, Environment
from rewind.bundle import extracted_bundle, write_bundle
from tests.test_workspace import docker_available


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.staging = self.root / 'staging'
        self.staging.mkdir()
        (self.staging / 'data.txt').write_text('saved state')
        (self.staging / 'empty').mkdir()
        self.archive = self.root / 'test.rewind'

    def test_roundtrip_preserves_bytes_and_empty_directories(self):
        write_bundle(self.staging, self.archive, {'kind': 'test'})
        with extracted_bundle(self.archive) as (staging, manifest):
            self.assertEqual(manifest['kind'], 'test')
            self.assertEqual((staging / 'data.txt').read_text(), 'saved state')
            self.assertTrue((staging / 'empty').is_dir())
        self.assertFalse(staging.exists())

    def test_existing_archive_is_not_overwritten(self):
        self.archive.write_text('original')
        with self.assertRaises(FileExistsError):
            write_bundle(self.staging, self.archive, {'kind': 'test'})
        self.assertEqual(self.archive.read_text(), 'original')

    def rewrite(self, mutate):
        original = self.root / 'original.tar'
        self.archive.rename(original)
        with tarfile.open(original) as source, tarfile.open(self.archive, 'w:gz') as dest:
            for item in source.getmembers():
                data = source.extractfile(item).read()
                item, data = mutate(item, data)
                item.size = len(data)
                dest.addfile(item, io.BytesIO(data))

    def test_corrupted_content_is_rejected_before_import_publishes(self):
        write_bundle(self.staging, self.archive, {'kind': 'checkpoint', 'checkpoint': 'checkpoint'})
        def mutate(item, data):
            if item.name == 'data.txt':
                data = b'other state'
            return item, data
        self.rewrite(mutate)
        output = self.root / 'imported'
        with patch('rewind.bundle._run') as docker:
            with self.assertRaisesRegex(ValueError, 'hash verification'):
                Checkpoint.import_bundle(self.archive, output)
        docker.assert_not_called()
        self.assertFalse(output.exists())

    def test_traversal_absolute_links_and_duplicate_entries_are_rejected(self):
        for name, kind in [('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                           ('C:/escape', tarfile.REGTYPE), ('C:escape', tarfile.REGTYPE),
                           ('link', tarfile.SYMTYPE), ('hard', tarfile.LNKTYPE),
                           ('fifo', tarfile.FIFOTYPE)]:
            with self.subTest(name=name):
                with tarfile.open(self.archive, 'w') as archive:
                    item = tarfile.TarInfo(name)
                    item.type = kind
                    item.linkname = '/tmp'
                    archive.addfile(item)
                with self.assertRaises(ValueError), extracted_bundle(self.archive):
                    pass
        with tarfile.open(self.archive, 'w') as archive:
            archive.addfile(tarfile.TarInfo('duplicate'))
            archive.addfile(tarfile.TarInfo('duplicate'))
        with self.assertRaisesRegex(ValueError, 'duplicate'), extracted_bundle(self.archive):
            pass
        self.assertFalse((self.root / 'escape').exists())

    def test_missing_inventory_entry_and_directory_traversal_are_rejected(self):
        for directories in (['../escape'], []):
            self.staging.joinpath('bundle.json').unlink(missing_ok=True)
            self.archive.unlink(missing_ok=True)
            write_bundle(self.staging, self.archive, {'kind': 'test'})
            def mutate(item, data):
                if item.name == 'bundle.json':
                    manifest = json.loads(data)
                    if directories:
                        manifest['directories'] = directories
                    else:
                        manifest['files'] = {}
                    data = json.dumps(manifest).encode()
                return item, data
            self.rewrite(mutate)
            with self.assertRaises(ValueError), extracted_bundle(self.archive):
                pass

    def test_export_refuses_symlink_assets(self):
        (self.staging / 'secret-link').symlink_to(self.root / 'elsewhere')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            write_bundle(self.staging, self.archive, {'kind': 'test'})
        self.assertFalse(self.archive.exists())

    def test_malformed_checkpoint_metadata_fails_without_publishing(self):
        saved = self.staging / 'checkpoint'
        saved.mkdir()
        (saved / 'meta.json').write_text('[]')
        write_bundle(self.staging, self.archive, {'kind': 'checkpoint', 'checkpoint': 'checkpoint'})
        output = self.root / 'imported'
        with patch('rewind.bundle._run') as docker:
            with self.assertRaisesRegex(ValueError, 'JSON object'):
                Checkpoint.import_bundle(self.archive, output)
        docker.assert_not_called()
        self.assertFalse(output.exists())


@unittest.skipUnless(docker_available(), 'requires Docker')
class DockerBundles(unittest.TestCase):
    def test_import_branches_after_original_project_files_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'original'
            source.mkdir()
            inputs = source / 'inputs$literal'
            inputs.mkdir()
            (inputs / 'empty').mkdir()
            (inputs / 'setting.txt').write_text('original input')
            compose = source / 'compose.yaml'
            compose.write_text('''services:
  default:
    image: debian:bookworm-slim
    command: [sleep, infinity]
    environment:
      LITERAL: 'a$$dollar'
    volumes:
      - state:/data
      - ./inputs$$literal:/inputs:ro
volumes:
  state:
''')
            archive = root / 'saved.rewind'
            messages = [{'role': 'user', 'content': 'continue from seven'}]
            with Environment.from_compose(compose, capture={'default': ['/data']}) as parent:
                self.assertEqual(parent.exec('default', 'printf 7 > /data/count')[0], 0)
                checkpoint = parent.checkpoint(source / 'checkpoint', messages=messages)
                self.assertEqual(parent.exec('default', 'printf 99 > /data/count')[0], 0)
                checkpoint.export(archive)
                self.assertEqual(parent.exec('default', 'cat /data/count'), (0, '99'))
            shutil.rmtree(source)
            with patch('rewind.bundle._run') as docker:
                restored = Checkpoint.import_bundle(archive, root / 'imported')
            docker.assert_not_called()
            self.assertEqual(restored.messages, messages)
            self.assertTrue(Path(restored.manifest['compose_file']).is_file())
            config = yaml.safe_load(Path(restored.manifest['compose_file']).read_text())
            images = json.loads((root / 'imported/environment/images.json').read_text())
            self.assertEqual(config['services']['default']['platform'], images['default']['platform'])
            with restored.branch() as branch:
                self.assertTrue(branch.restore_report['ok'])
                self.assertEqual(branch.exec('default', 'cat /data/count'), (0, '7'))
                self.assertEqual(branch.exec('default', 'cat /inputs/setting.txt'), (0, 'original input'))
                self.assertEqual(branch.exec('default', 'test -d /inputs/empty')[0], 0)
                self.assertEqual(branch.exec('default', 'printf %s "$LITERAL"'), (0, 'a$dollar'))
            with self.assertRaises(FileExistsError):
                Checkpoint.import_bundle(archive, root / 'imported')
            self.assertEqual(Checkpoint.load(restored.path).messages, messages)


if __name__ == '__main__':
    unittest.main()
