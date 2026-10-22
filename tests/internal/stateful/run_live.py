"""Opt-in live-provider check; run from the repository root with PYTHONPATH=."""
import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from rewind import Checkpoint
from tests.test_stateful import ScriptedAgent, StatefulDocker, StatefulLifecycle

TOOL = {'type': 'function', 'function': {
    'name': 'environment',
    'description': 'Use the job service. inspect shows config, jobs and events; wait waits for a job.',
    'parameters': {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['inspect', 'configure', 'submit', 'enable', 'wait']},
        'id': {'type': 'string', 'description': 'Exact job identifier for submit, enable, or wait.'},
        'value': {'type': 'string', 'description': 'Configuration string, only for configure.'},
        'enabled': {'type': 'boolean', 'description': 'Whether a submitted job may run.'},
    }, 'required': ['action'], 'additionalProperties': False},
}}


def continue_agent(agent, model, messages, key, stop_at_queue=False):
    messages = copy.deepcopy(messages)
    for _ in range(15):
        request = Request('https://openrouter.ai/api/v1/chat/completions',
            data=json.dumps({'model': model, 'messages': messages, 'tools': [TOOL],
                             'parallel_tool_calls': False, 'max_tokens': 1500}).encode(),
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        with urlopen(request, timeout=90) as response:
            body = json.load(response)
        response = body['choices'][0]['message']
        # Keep portable text and function calls, not provider-specific reasoning.
        message = {k: response[k] for k in ('role', 'content', 'tool_calls') if k in response}
        calls = message.get('tool_calls') or []
        if len(calls) > 1:
            raise RuntimeError('live check requires sequential tool calls')
        messages.append(message)
        if not calls:
            if stop_at_queue:
                raise AssertionError('agent finished before the checkpoint boundary')
            return messages
        tool = calls[0]
        assert tool['function']['name'] == 'environment'
        args = json.loads(tool['function']['arguments'])
        action = args.pop('action')
        if action == 'inspect':
            result = agent.state()
            # Authentication secrets belong to the adapter, not the model.
            result.pop('token', None)
        elif action == 'wait':
            agent.done(args['id'])
            result = {'done': args['id']}
        else:
            if action not in ('configure', 'submit', 'enable'):
                raise ValueError('unknown action')
            result = agent.step('/' + action, **args)
        messages.append({'role': 'tool', 'tool_call_id': tool['id'],
                         'content': json.dumps(result)})
        print(f'{model}: {action} {args}', flush=True)
        if stop_at_queue and action == 'submit' and args.get('id') == 'queued':
            state = agent.state()
            assert state['jobs']['completed']['status'] == 'done'
            assert state['jobs']['queued'] == {'status': 'pending', 'enabled': False}
            return messages
    raise AssertionError('agent exceeded tool-call budget')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-model', required=True)
    parser.add_argument('--branch-model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    load_dotenv()
    key = os.environ['OPENROUTER_API_KEY']
    args.output.mkdir(parents=True, exist_ok=False)
    harness = StatefulDocker()
    messages = [
        {'role': 'system', 'content': 'Use the environment tool sequentially to perform the task. '
         'Inspect existing state before changes. Continue existing work without recreating jobs.'},
        {'role': 'user', 'content': 'Submit a job with the exact id "completed" and wait for it to finish. Then submit '
         'a job with the exact id "queued" and enabled=false. After that enable the job '
         '"queued", wait for completion, and finish. Use these exact ids, with no suffixes.'},
    ]
    with ExitStack() as stack:
        parent = harness.parent(stack)
        prefix = continue_agent(ScriptedAgent(parent), args.parent_model, messages, key, True)
        checkpoint = parent.checkpoint(args.output / 'checkpoint', messages=prefix)
        expected = ScriptedAgent(parent).state()
        results = {}
        for label, model in [('parent', args.parent_model), ('same-model', args.parent_model),
                             ('different-model', args.branch_model)]:
            environment = parent if label == 'parent' else Checkpoint.load(checkpoint.path).branch(
                lifecycle=StatefulLifecycle())
            if environment is not parent:
                stack.callback(environment.down)
            agent = ScriptedAgent(environment)
            assert agent.state() == expected
            transcript = continue_agent(agent, model, prefix + [
                {'role': 'user', 'content': (checkpoint.restore_notice if label != 'parent' else '') +
                 f' Continue the task, but first set configuration to {label}. Do not resubmit jobs.'}
            ], key)
            state = agent.state()
            assert state['config'] == label
            assert state['jobs']['queued']['status'] == 'done'
            assert sorted(state['events']) == ['completed', 'queued']
            receipt = json.loads(environment.exec('worker', ['cat', '/data/queued.json'])[1])
            assert receipt['config'] == label
            assert transcript[:len(prefix)] == prefix
            (args.output / (label + '.json')).write_text(json.dumps(transcript, indent=2))
            results[label] = {'model': model, 'prefix_matches': True, 'config': state['config'],
                              'events': state['events']}
        assert ScriptedAgent(parent).state()['config'] == 'parent'
        (args.output / 'result.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
