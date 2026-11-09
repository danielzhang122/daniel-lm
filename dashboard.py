"""Read-only dashboard for local training logs. No extra dependencies."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


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
    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/api/runs':
            body = json.dumps(snapshot(), allow_nan=False).encode()
            content_type = 'application/json'
        elif path in ('/', '/index.html', '/app.js', '/style.css'):
            name = 'index.html' if path == '/' else path[1:]
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
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'daniellm: http://127.0.0.1:{args.port} (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
