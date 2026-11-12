"""Live petri viewer: serves viewer.html and computes traces on demand on the Mac's GPU.

  env/result/bin/python3 petri/server.py [port]      then open http://127.0.0.1:8765/

Listens on 127.0.0.1 only. GET /api/prompts lists the fixed samples; GET /api/trace?name=<sample>
returns a sample's trace; POST /api/trace with {"text": "..."} traces any prompt (2 to 96 tokens).
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace import HERE, SKELETON, Tracer  # noqa: E402  (petri/trace.py, not the stdlib module: HERE comes first)

PROMPTS = json.load(open(os.path.join(HERE, 'prompts.json')))['prompts']
tracer = Tracer()
lock = threading.Lock()
cache = {}

def sample_trace(name):
    if name not in cache:
        p = next(p for p in PROMPTS if p['name'] == name)
        with lock:
            cache[name] = tracer.run(p['text'], p['name'], p['domain'], 'petri/prompts.json (fixed sample)')
    return cache[name]


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def fail(self, code, msg):
        self.send(code, json.dumps({'error': msg}).encode(), 'application/json')

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ('/', '/index.html'):
            page = SKELETON.replace('%BODY%', open(os.path.join(HERE, 'viewer.html')).read())
            return self.send(200, page.encode(), 'text/html; charset=utf-8')
        if u.path.startswith('/vendor/') and '..' not in u.path:
            f = os.path.join(HERE, u.path.lstrip('/'))
            if os.path.isfile(f):
                return self.send(200, open(f, 'rb').read(), 'text/javascript')
        if u.path == '/api/prompts':
            lst = [{'name': p['name'], 'domain': p['domain'], 'text': p['text'],
                    'url': f"/api/trace?name={p['name']}"} for p in PROMPTS]
            return self.send(200, json.dumps({'mode': 'live', 'prompts': lst, 'weights': '/api/weights'}).encode(),
                             'application/json')
        if u.path == '/api/weights':
            if 'weights' not in cache:
                with lock:
                    cache['weights'] = tracer.weights()
            return self.send(200, cache['weights'], 'application/octet-stream')
        if u.path == '/api/trace':
            name = parse_qs(u.query).get('name', [''])[0]
            if not any(p['name'] == name for p in PROMPTS):
                return self.fail(404, f'no sample called {name!r}')
            return self.send(200, sample_trace(name), 'application/octet-stream')
        self.fail(404, 'not found')

    def do_POST(self):
        if urlparse(self.path).path != '/api/trace':
            return self.fail(404, 'not found')
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            with lock:
                data = tracer.run(str(body.get('text', '')))
        except ValueError as e:
            return self.fail(400, str(e))
        self.send(200, data, 'application/octet-stream')

    def log_message(self, fmt, *args):
        sys.stderr.write('petri: ' + fmt % args + '\n')


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'petri viewer on http://127.0.0.1:{port}/  (blocks: {tracer.blocks_meta["kind"]})', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
