"""Real Docker regressions for application restore and non-Inspect continuation."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import uuid

from rewind import Checkpoint, Environment
from tests.test_workspace import docker_available

FIXTURE = Path(__file__).parent / 'internal' / 'stateful' / 'compose.yaml'
CAPTURE = {name: ['/data'] for name in ('agent', 'api', 'worker')}


def call(environment, path, **body):
    rc, out = environment.exec('agent', ['python', '/app.py', 'request', path, json.dumps(body)], timeout=20)
    if rc:
        raise RuntimeError(f'{path}: {out}')
    return json.loads(out)


def eventually(check, timeout=20):
    deadline = time.monotonic() + timeout
    while True:
        try:
            value = check()
            if value:
                return value
        except RuntimeError:
            pass
        if time.monotonic() >= deadline:
            raise AssertionError('service did not reach expected state')
        time.sleep(0.1)


class StatefulLifecycle:
    name = 'stateful-test-v1'

    def before_checkpoint(self, environment):
        call(environment, '/quiesce')

    def after_checkpoint(self, environment):
        call(environment, '/resume')

    def after_restore(self, environment, checkpoint):
        eventually(lambda: call(environment, '/health'))
        restored = call(environment, '/restore')
        rc, token = environment.exec('agent', ['cat', '/data/token'])
        if rc or restored != call(environment, '/state', token=token):
            raise RuntimeError('application state or authentication did not restore')
        call(environment, '/resume')


class ScriptedAgent:
    """Caller-owned transcript; a branch continues after the last saved tool."""
    def __init__(self, environment, messages=()):
        self.environment = environment
        self.messages = copy.deepcopy(list(messages))
        self.token = environment.exec('agent', ['cat', '/data/token'])[1]

    def step(self, path, **body):
        result = call(self.environment, path, token=self.token, **body)
        self.messages.append({'tool': path, 'arguments': body, 'result': result})
        return result

    def state(self):
        return call(self.environment, '/state', token=self.token)

    def done(self, job):
        return eventually(lambda: self.state()['jobs'].get(job, {}).get('status') == 'done')


@unittest.skipUnless(docker_available(), 'requires Docker')
class StatefulDocker(unittest.TestCase):
    def parent(self, stack, lifecycle=True):
        env = stack.enter_context(Environment.from_compose(
            FIXTURE, capture=CAPTURE, lifecycle=StatefulLifecycle() if lifecycle else None))
        eventually(lambda: call(env, '/health'))
        token = call(env, '/init')['token']
        rc, out = env.exec('agent', ['python', '-c',
            "from pathlib import Path; import sys; Path('/data/token').write_text(sys.argv[1])", token])
        self.assertEqual(rc, 0, out)
        call(env, '/resume')
        return env

    def test_background_jobs_authentication_and_nested_continuation(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            parent = self.parent(stack)
            agent = ScriptedAgent(parent)
            agent.step('/submit', id='completed')
            agent.done('completed')
            agent.step('/submit', id='queued', enabled=False)
            agent.step('/submit', id='inflight')
            eventually(lambda: agent.state()['jobs']['inflight']['status'] == 'active')
            checkpoint = parent.checkpoint(Path(tmp) / 'first', messages=agent.messages,
                                           metadata={'scorer_start': 123, 'run_id': 'parent'})
            expected = agent.state()
            self.assertEqual(expected['jobs']['inflight']['status'], 'done')
            prefix = copy.deepcopy(agent.messages)
            agent.step('/configure', value='parent-only')
            agent.step('/enable', id='queued')
            agent.done('queued')

            branches = []
            for label in ('A', 'B'):
                branch = checkpoint.branch(lifecycle=StatefulLifecycle())
                stack.callback(branch.down)
                continuation = ScriptedAgent(branch, Checkpoint.load(checkpoint.path).messages)
                self.assertEqual(continuation.messages, prefix)
                self.assertEqual(continuation.state(), expected)
                continuation.step('/configure', value=label)
                continuation.step('/enable', id='queued')
                continuation.done('queued')
                continuation.step('/submit', id=label)
                continuation.done(label)
                branches.append((branch, continuation))
            for label, (branch, continuation) in zip(('A', 'B'), branches):
                state = continuation.state()
                self.assertEqual(state['config'], label)
                self.assertEqual(state['events'].count('completed'), 1)
                self.assertEqual(state['events'].count('inflight'), 1)
                self.assertNotIn('B' if label == 'A' else 'A', state['jobs'])
                rc, receipt = branch.exec('worker', ['cat', '/data/queued.json'])
                self.assertEqual(rc, 0)
                self.assertEqual(json.loads(receipt)['config'], label)
            self.assertEqual(agent.state()['config'], 'parent-only')
            self.assertNotIn('A', agent.state()['jobs'])
            second = branches[1][0].checkpoint(Path(tmp) / 'second',
                                              messages=branches[1][1].messages)
            nested = second.branch(lifecycle=StatefulLifecycle())
            stack.callback(nested.down)
            resumed = ScriptedAgent(nested, second.messages)
            self.assertEqual(resumed.state(), branches[1][1].state())
            resumed.step('/submit', id='nested')
            resumed.done('nested')
            self.assertNotIn('nested', branches[1][1].state()['jobs'])
            self.assertEqual(checkpoint.metadata['scorer_start'], 123)

    def test_restore_requires_adapter_and_readiness_failure_cleans_branch(self):
        class RejectRestore(StatefulLifecycle):
            def after_restore(self, environment, checkpoint):
                super().after_restore(environment, checkpoint)
                raise RuntimeError('application readiness rejected')

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            parent = self.parent(stack)
            checkpoint = parent.checkpoint(Path(tmp) / 'checkpoint')
            with self.assertRaisesRegex(ValueError, 'requires lifecycle adapter'):
                checkpoint.branch()
            bundle = checkpoint.export(Path(tmp) / 'saved.tar.gz')
            imported = Checkpoint.import_bundle(bundle, Path(tmp) / 'imported')
            with self.assertRaisesRegex(ValueError, 'requires lifecycle adapter'):
                imported.branch()
            restored = imported.branch(lifecycle=StatefulLifecycle())
            stack.callback(restored.down)
            self.assertEqual(ScriptedAgent(restored).state(), ScriptedAgent(parent).state())
            project = 'rewind_reject_' + uuid.uuid4().hex[:10]
            with self.assertRaisesRegex(RuntimeError, 'application readiness rejected'):
                checkpoint.branch(project=project, lifecycle=RejectRestore())
            for command in (['docker', 'ps', '-aq'], ['docker', 'volume', 'ls', '-q'],
                            ['docker', 'network', 'ls', '-q']):
                resources = subprocess.check_output(command + [
                    '--filter', f'label=com.docker.compose.project={project}'], text=True)
                self.assertEqual(resources.strip(), '')
            self.assertTrue(ScriptedAgent(parent).state()['token'])

    def test_filesystem_verification_alone_does_not_restore_application(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            parent = self.parent(stack, lifecycle=False)
            call(parent, '/quiesce')
            checkpoint = parent.checkpoint(Path(tmp) / 'without-adapter')
            call(parent, '/resume')
            branch = checkpoint.branch()
            stack.callback(branch.down)
            eventually(lambda: call(branch, '/health'))
            self.assertTrue(branch.restore_report['ok'])
            with self.assertRaisesRegex(RuntimeError, '403'):
                ScriptedAgent(branch).state()


if __name__ == '__main__':
    unittest.main()
