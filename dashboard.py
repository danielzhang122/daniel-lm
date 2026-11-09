"""Local training dashboard and CPU text-generation playground."""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import library
import sample_store

ROOT = Path(__file__).resolve().parent
GENERATION_LOCK = threading.Lock()


class DashboardServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        # Windows SO_REUSEADDR can let two servers bind the same port.
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def generation_command(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected a JSON object.')
    run = payload.get('run', '')
    prompt = payload.get('prompt', '')
    messages=payload.get('messages')
    if messages is not None:
        if not isinstance(messages,list) or not 1<=len(messages)<=21 or any(not isinstance(m,dict) or not isinstance(m.get('content'),str) or m.get('role') not in ('user','assistant') for m in messages):
            raise ValueError('Invalid conversation')
        prompt=messages[-1]['content']
    tokens = payload.get('tokens', 80)
    temperature = payload.get('temperature', 0.8)
    if not isinstance(run, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', run):
        raise ValueError('Select a valid run.')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
        raise ValueError('Enter a prompt between 1 and 2,000 characters.')
    if type(tokens) is not int or not 1 <= tokens <= 200:
        raise ValueError('Choose between 1 and 200 tokens.')
    if type(temperature) not in (int, float) or not 0.1 <= temperature <= 2:
        raise ValueError('Choose a temperature between 0.1 and 2.')
    folder = ROOT / 'runs' / run
    if not (folder / 'latest.pt').is_file():
        raise ValueError('This run has no saved checkpoint yet.')
    config = read_json(folder / 'run.json', {})
    data = Path(config.get('data', ROOT / 'data' / run))
    command = [sys.executable, str(ROOT / 'generate.py'), '--checkpoint', str(folder / 'latest.pt'),
            '--tokenizer', str(data / 'tokenizer.json'), '--prompt', prompt, '--tokens', str(tokens),
            '--temperature', str(temperature), '--device', 'cpu', '--json']
    if messages is not None:
        if config.get('phase')!='dialogue': raise ValueError('Choose a dialogue-trained run; current pretraining runs are text completion only.')
        command += ['--messages-json',json.dumps(messages)]
    return command


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def snapshot():
    runs = []
    for folder in sorted((ROOT / 'runs').glob('*')):
        metrics = folder / 'metrics.jsonl'
        if not folder.is_dir() or not metrics.exists():
            continue
        rows = []
        for line in metrics.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict) and isinstance(row.get('step'), int):
                    rows.append(row)
            except ValueError:
                pass  # The trainer may be halfway through writing its last line.
        config = read_json(folder / 'run.json', {})
        data_path = Path(config.get('data', ROOT / 'data' / folder.name))
        runs.append(dict(name=folder.name, rows=rows, config=config,
                         dataset=read_json(data_path / 'metadata.json', {}),
                         updated=metrics.stat().st_mtime,
                         checkpoint=(folder / 'latest.pt').exists()))
    return dict(runs=runs, project=str(ROOT))


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/api/generate':
            self.send_json(404, {'error': 'Not found'})
            return
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        host = self.headers.get('Host', '')
        if host not in allowed or self.headers.get('Origin', 'http://' + host) != 'http://' + host:
            self.send_json(403, {'error': 'Use the local dashboard to generate text.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16000:
                raise ValueError('Request is too large or empty.')
            if self.headers.get_content_type() != 'application/json':
                raise ValueError('Expected application/json.')
            payload = json.loads(self.rfile.read(length))
            command = generation_command(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            self.send_json(400, {'error': str(exc)})
            return
        if not GENERATION_LOCK.acquire(blocking=False):
            self.send_json(409, {'error': 'A sample is already generating. Try again when it finishes.'})
            return
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                    timeout=120, cwd=ROOT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            if result.returncode:
                message = 'Shorten your message; it exceeds the model context budget.' if 'too long for this model' in result.stderr else 'Generation failed. Check the checkpoint and matching tokenizer.'
                self.send_json(500, {'error': message})
            else:
                generated = json.loads(result.stdout)
                generated['sample_id'] = sample_store.save(payload['run'],payload.get('prompt') or payload['messages'][-1]['content'],generated,
                    payload.get('temperature',.8),payload.get('tokens',80))
                self.send_json(200, generated)
        except subprocess.TimeoutExpired:
            self.send_json(504, {'error': 'Generation took too long. Try fewer tokens or a shorter prompt.'})
        except (OSError, ValueError):
            self.send_json(500, {'error': 'Could not run generation.'})
        finally:
            GENERATION_LOCK.release()

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == '/api/samples':
            self.send_json(200, {'samples':sample_store.recent()})
            return
        if path == '/api/library':
            self.send_json(200, {'status':library.status(),'results':library.search(query.get('q',[''])[0][:300])})
            return
        if path == '/api/article':
            try:
                item=library.article(int(query.get('id',['0'])[0]))
            except ValueError:
                item=None
            self.send_json(200 if item else 404,item or {'error':'Article not found'})
            return
        if path == '/api/runs':
            body = json.dumps(snapshot(), allow_nan=False).encode()
            content_type = 'application/json'
        elif path in ('/', '/index.html', '/app.js', '/style.css', '/play', '/play.html', '/play.js', '/library', '/library.html', '/library.js', '/samples', '/samples.html', '/samples.js','/chat','/chat.html','/chat.js'):
            name = {'/':'index.html','/play':'play.html','/library':'library.html','/samples':'samples.html','/chat':'chat.html'}.get(path,path[1:])
            body = (ROOT / 'dashboard' / name).read_bytes()
            content_type = {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[name.rsplit('.', 1)[1]]
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    try:
        server = DashboardServer(('127.0.0.1', args.port), Handler)
    except OSError as exc:
        raise SystemExit(f'Cannot start dashboard on port {args.port}: {exc}. Stop the existing dashboard or use --port 8766.')
    print(f'daniellm: http://127.0.0.1:{args.port} (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
