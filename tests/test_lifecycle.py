"""Failure and ownership contracts for caller-supplied lifecycle adapters."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from rewind import Environment


class LifecycleContracts(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.compose = self.root / 'compose.yaml'
        self.compose.write_text('services: {}\n')
        self.adapter = Mock()
        self.adapter.name = 'test-v1'
        self.environment = Environment.from_compose(
            self.compose, capture={'agent': ['/data']}, lifecycle=self.adapter)
        self.environment._started = True
        self.output = self.root / 'checkpoint'

    def test_partial_preparation_is_resumed_and_never_published(self):
        self.adapter.before_checkpoint.side_effect = RuntimeError('flush failed')
        with patch('rewind.workspace.snapshot') as snapshot:
            with self.assertRaisesRegex(RuntimeError, 'flush failed'):
                self.environment.checkpoint(self.output)
        snapshot.assert_not_called()
        self.adapter.after_checkpoint.assert_called_once_with(self.environment)
        self.assertFalse(self.output.exists())

    def test_snapshot_failure_resumes_and_preserves_original_error(self):
        self.adapter.after_checkpoint.side_effect = RuntimeError('resume failed')
        with patch.object(self.environment._engine, 'container_id', return_value='id'), \
             patch('rewind.workspace._run', return_value=SimpleNamespace(returncode=0)), \
             patch('rewind.workspace.snapshot', side_effect=RuntimeError('copy failed')):
            with self.assertRaisesRegex(RuntimeError, 'copy failed') as raised:
                self.environment.checkpoint(self.output)
        self.assertIn('resume failed', raised.exception.__notes__[0])
        self.adapter.after_checkpoint.assert_called_once()
        self.assertFalse(self.output.exists())

    def test_resume_failure_does_not_publish_and_hooks_surround_snapshot(self):
        order = []
        self.adapter.before_checkpoint.side_effect = lambda env: order.append('prepare')
        def resume(env):
            order.append('resume')
            raise RuntimeError('resume failed')
        self.adapter.after_checkpoint.side_effect = resume
        def snapshot(engine, directory, details):
            order.append('snapshot')
            self.assertEqual(details['lifecycle'], 'test-v1')
            directory.mkdir()
            return details
        with patch.object(self.environment._engine, 'container_id', return_value='id'), \
             patch('rewind.workspace._run', return_value=SimpleNamespace(returncode=0)), \
             patch('rewind.workspace.snapshot', side_effect=snapshot):
            with self.assertRaisesRegex(RuntimeError, 'resume failed'):
                self.environment.checkpoint(self.output)
        self.assertEqual(order, ['prepare', 'snapshot', 'resume'])
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.checkpoint-*')))

    def test_saved_handle_requires_explicit_matching_adapter(self):
        self.environment._owned = True
        handle = self.environment.save(self.root / 'handle.json')
        with self.assertRaisesRegex(ValueError, 'requires lifecycle'):
            Environment.load(handle)
        wrong = Mock()
        wrong.name = 'other-v1'
        with self.assertRaisesRegex(ValueError, 'requires lifecycle'):
            Environment.load(handle, lifecycle=wrong)
        restored = Environment.load(handle, lifecycle=self.adapter)
        self.assertIs(restored.lifecycle, self.adapter)
        self.assertEqual(restored.project, self.environment.project)

    def test_existing_destination_does_not_quiesce_services(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.environment.checkpoint(self.output)
        self.adapter.before_checkpoint.assert_not_called()


if __name__ == '__main__':
    unittest.main()
