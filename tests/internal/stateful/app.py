"""Internal fixture: deliberately volatile API state and an idempotent worker."""
import json
from pathlib import Path
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(path, body=None):
    req = Request('http://api:8080' + path, data=json.dumps(body or {}).encode(),
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=15) as response:
        return json.load(response)


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def api():
    # Startup never hydrates implicitly or permits background work. The adapter
    # must restore and check state before releasing workers.
    state = {}
    paused = True
    condition = threading.Condition()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            nonlocal state, paused
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            result, status = {}, 200
            with condition:
                if self.path in ('/state', '/configure', '/submit', '/enable') and (
                    not state or body.get('token') != state['token']
                ):
                    status, result = 403, {'error': 'invalid token'}
                elif self.path == '/health':
                    result = {'ready': True}
                elif self.path == '/init':
                    state = {'token': secrets.token_hex(16), 'config': 'original',
                             'jobs': {}, 'events': []}
                    result = state
                elif self.path == '/state':
                    result = state
                elif self.path == '/configure':
                    state['config'] = body['value']
                elif self.path == '/submit':
                    if paused:
                        status = 409
                    else:
                        state['jobs'].setdefault(body['id'], {
                            'status': 'pending', 'enabled': body.get('enabled', True)})
                elif self.path == '/enable':
                    state['jobs'][body['id']]['enabled'] = True
                elif self.path == '/claim':
                    if not paused:
                        for key, job in state['jobs'].items():
                            if job['enabled'] and job['status'] == 'pending':
                                job['status'] = 'active'
                                result = {'id': key, 'config': state['config']}
                                break
                elif self.path == '/complete':
                    job = state['jobs'][body['id']]
                    if job['status'] != 'done':
                        job['status'] = 'done'
                        state['events'].append(body['id'])
                    condition.notify_all()
                elif self.path == '/quiesce':
                    paused = True
                    deadline = time.monotonic() + 10
                    while any(j['status'] == 'active' for j in state['jobs'].values()):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            status = 503
                            break
                        condition.wait(remaining)
                    if status == 200:
                        atomic_json(Path('/data/state.json'), state)
                        result = state
                elif self.path == '/resume':
                    paused = False
                elif self.path == '/restore':
                    restored = json.loads(Path('/data/state.json').read_text())
                    if not restored.get('token') or any(
                        j['status'] == 'active' for j in restored['jobs'].values()
                    ):
                        status = 409
                    else:
                        state = restored
                        result = state
                else:
                    status = 404
                # Serialize under the lock: never expose half of a worker update.
                payload = json.dumps(result).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    ThreadingHTTPServer(('0.0.0.0', 8080), Handler).serve_forever()


def worker():
    while True:
        try:
            job = request('/claim')
            if job:
                time.sleep(0.6)
                receipt = Path('/data') / (job['id'] + '.json')
                if not receipt.exists():
                    atomic_json(receipt, job)
                request('/complete', {'id': job['id']})
        except (URLError, TimeoutError):
            pass
        time.sleep(0.05)


if __name__ == '__main__':
    if sys.argv[1] == 'api':
        api()
    elif sys.argv[1] == 'worker':
        worker()
    elif sys.argv[1] == 'request':
        try:
            print(json.dumps(request(sys.argv[2], json.loads(sys.argv[3]))))
        except HTTPError as exc:
            print(json.dumps({'status': exc.code}))
            sys.exit(1)
