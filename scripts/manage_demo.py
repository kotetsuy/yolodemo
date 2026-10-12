# SPDX-License-Identifier: AGPL-3.0-only
"""Background startup and graceful shutdown, scoped to this workspace."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.cache/run'
STATE = RUN / 'state.json'
LOG = RUN / 'demo.log'


def process(pid):
    try:
        proc = Path('/proc') / str(pid)
        fields = (proc/'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), birth=fields[19], parent=int(fields[1]),
                    cwd=(proc/'cwd').resolve(strict=True),
                    args=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def ours(p):
    return p and p['cwd'] == ROOT and (
        ('-m' in p['args'] and 'app.server' in p['args']) or
        str(ROOT/'scripts/run_demo.sh') in p['args'])


def servers():
    found = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            p = process(entry.name)
            if ours(p):
                found[p['pid']] = p
    if STATE.exists():
        try:
            saved = json.loads(STATE.read_text())
            p = process(saved['pid'])
            if ours(p) and p['birth'] == saved['birth']:
                found[p['pid']] = p
        except (OSError, ValueError, KeyError):
            pass
    return list(found.values())


def same(p):
    current = process(p['pid'])
    return current is not None and current['birth'] == p['birth']


def send(p, sig):
    if same(p):
        try:
            os.kill(p['pid'], sig)
        except ProcessLookupError:
            pass


def port(args):
    for i, value in enumerate(args):
        if value.startswith('--port='):
            return value.split('=', 1)[1]
        if value == '--port' and i+1 < len(args):
            return args[i+1]
    return '8000'


def start(args):
    existing = servers()
    if existing:
        print(f'既に起動しています (PID {existing[0]["pid"]})。')
        print(f'http://127.0.0.1:{port(existing[0]["args"])}')
        return
    if '--help' in args or '-h' in args:
        subprocess.run([sys.executable, '-m', 'app.server', '--help'], cwd=ROOT, check=True)
        return
    with LOG.open('ab', buffering=0) as log:
        proc = subprocess.Popen(['bash', str(ROOT/'scripts/run_demo.sh'), *args],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    p = process(proc.pid)
    if p is None:
        raise RuntimeError(f'起動に失敗しました。ログ: {LOG}')
    STATE.write_text(json.dumps(dict(pid=p['pid'], birth=p['birth']))+'\n')
    # Catch CLI and bootstrap failures; model preparation continues in background.
    for _ in range(10):
        if proc.poll() is not None:
            STATE.unlink(missing_ok=True)
            raise RuntimeError(f'起動処理が終了しました ({proc.returncode})。ログ: {LOG}')
        time.sleep(.1)
    print(f'起動しました (PID {proc.pid})。モデル準備は画面で確認できます。')
    print(f'http://127.0.0.1:{port(args)}\nログ: {LOG}\n停止: ./stop_all.sh')


def stop(args):
    if args:
        raise ValueError('停止は ./stop_all.sh （引数なし）で実行してください。')
    roots = servers()
    if not roots:
        STATE.unlink(missing_ok=True)
        print('既に停止しています。')
        return
    descendants = {p['pid']: p for p in roots}
    candidates = [p for entry in Path('/proc').iterdir() if entry.name.isdecimal()
                  if (p := process(entry.name)) is not None]
    while True:
        # Chrome may use the user's ordinary profile and other tabs. Leave its
        # process tree running; only stop the application and its compute jobs.
        added = [p for p in candidates if p['parent'] in descendants and p['pid'] not in descendants
                 and Path(p['args'][0]).name not in ('chrome', 'google-chrome', 'chromium', 'chromium-browser')]
        if not added:
            break
        descendants.update((p['pid'], p) for p in added)
    buses = set()
    for p in descendants.values():
        for i, arg in enumerate(p['args'][:-1]):
            if arg == '--bus':
                path = Path(p['args'][i+1])
                if path.parent == Path('/dev/shm') and path.name.startswith('yolodemo-'):
                    buses.add(path)
    for p in roots:
        send(p, signal.SIGTERM)
    deadline = time.monotonic()+30
    while time.monotonic() < deadline and any(same(p) for p in descendants.values()):
        time.sleep(.1)
    remaining = [p for p in descendants.values() if same(p)]
    if remaining:
        for p in remaining:
            send(p, signal.SIGTERM)
        deadline = time.monotonic()+3
        while time.monotonic() < deadline and any(same(p) for p in remaining):
            time.sleep(.1)
        for p in remaining:
            send(p, signal.SIGKILL)
        deadline = time.monotonic()+3
        while time.monotonic() < deadline and any(same(p) for p in remaining):
            time.sleep(.1)
    if any(same(p) for p in descendants.values()):
        raise RuntimeError('停止できないプロセスがあります。ログを確認してください。')
    for path in buses:
        path.unlink(missing_ok=True)
    STATE.unlink(missing_ok=True)
    print('停止しました（サーバー・推論worker・共有メモリ）。')


if __name__ == '__main__':
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'control.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            {'start': start, 'stop': stop}[sys.argv[1]](sys.argv[2:])
        except (RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
