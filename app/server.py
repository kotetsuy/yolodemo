# SPDX-License-Identifier: AGPL-3.0-only
"""Local web UI and JSON/MJPEG endpoints using Python's HTTP server."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import shutil
import signal
import subprocess
import threading
from urllib.parse import urlparse
from app.coordinator import Coordinator
from app.model_utils import ROOT
from app.cameras import DEFAULT_CONFIG
from app.source import source_archive


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['auto', 'playlist'], default='auto')
    p.add_argument('--camera')
    p.add_argument('--camera-config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--camera-width', type=int)
    p.add_argument('--camera-height', type=int)
    p.add_argument('--camera-fps', type=float)
    p.add_argument('--playlist', type=Path, default=ROOT / 'videos/playlist.m3u8')
    p.add_argument('--backend', choices=['gpu', 'npu'], default='gpu')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--display-width', type=int, default=1920)
    p.add_argument('--no-browser', action='store_true')
    return p


def handler(coordinator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            if args and str(args[1] if len(args)>1 else '') not in ('200',):
                logging.info(fmt, *args)

        def reply(self, data, mime, code=200):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def json(self, value, code=200):
            self.reply(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8', code)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == '/api/status':
                    self.json(coordinator.status())
                elif path == '/api/cameras':
                    self.json(coordinator.capture.status())
                elif path == '/stream.mjpg':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                    self.connection.settimeout(2)
                    sequence = -1
                    while not coordinator.stop.is_set():
                        new_sequence, image = coordinator.next_jpeg(sequence)
                        if image is None or sequence == new_sequence:
                            continue
                        sequence = new_sequence
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(image)).encode()+b'\r\n\r\n'+image+b'\r\n')
                        self.wfile.flush()
                elif path == '/frame.jpg':
                    _, image = coordinator.next_jpeg(-1)
                    if image is None:
                        self.json(dict(error='No current detection frame'), 503)
                    else:
                        self.reply(image, 'image/jpeg')
                elif path in ('/', '/static/style.css', '/static/app.js'):
                    filename, mime = {
                        '/': ('index.html', 'text/html; charset=utf-8'),
                        '/static/style.css': ('style.css', 'text/css; charset=utf-8'),
                        '/static/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                    }[path]
                    self.reply((ROOT/'app/static'/filename).read_bytes(), mime)
                elif path == '/license':
                    self.reply((ROOT/'LICENSE').read_bytes(), 'text/plain; charset=utf-8')
                elif path == '/source.zip':
                    self.reply(source_archive(ROOT), 'application/zip')
                elif path == '/credits':
                    self.reply((ROOT/'videos/SOURCES.md').read_bytes(), 'text/plain; charset=utf-8')
                elif path == '/favicon.ico':
                    self.reply(b'', 'image/x-icon', 204)
                else:
                    self.json(dict(error='Not found'), 404)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ('/api/backend', '/api/camera'):
                self.json(dict(error='Not found'), 404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(length))
                if path == '/api/backend':
                    coordinator.request(data['backend'])
                else:
                    coordinator.capture.request_camera(data['camera'])
                self.json(dict(accepted=True, **coordinator.status()))
            except (ValueError, KeyError, TypeError) as exc:
                self.json(dict(error=str(exc)), 409)

    return Handler


def main():
    args = parser().parse_args()
    if not 1 <= args.port <= 65535 or args.display_width < 64:
        raise SystemExit('Invalid port or display width')
    args.playlist = args.playlist.expanduser().resolve()
    coordinator = Coordinator(args)
    try:
        http = ThreadingHTTPServer(('127.0.0.1', args.port), handler(coordinator))
    except Exception:
        coordinator.bus.close()
        coordinator.path.unlink(missing_ok=True)
        raise
    http.daemon_threads = True
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    def stop(*_):
        threading.Thread(target=http.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        coordinator.start()
        url = f'http://127.0.0.1:{args.port}'
        print(f'YOLO demo: {url}\nResults: {coordinator.folder}', flush=True)
        if not args.no_browser:
            chrome = shutil.which('google-chrome') or shutil.which('chromium')
            if chrome:
                subprocess.Popen([chrome, '--new-window', url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                logging.warning('Chrome not found. Open %s manually.', url)
        http.serve_forever(poll_interval=.2)
    finally:
        http.server_close()
        coordinator.close()


if __name__ == '__main__':
    main()
