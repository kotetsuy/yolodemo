# SPDX-License-Identifier: AGPL-3.0-only
"""Capture/worker lifecycle, frame matching, rendering, and live statistics."""
from collections import OrderedDict
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time
import uuid
import cv2
import numpy as np
from app.capture import Capture
from app.cameras import load_config
from app.ipc import InputBus
from app.metrics import FpsMeter
from app.model_utils import ROOT, YOLOFAST, write_json
from app.preprocess import preprocess, restore


class Coordinator:
    def __init__(self, args):
        cv2.setNumThreads(1)
        self.args = args
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.stop = threading.Event()
        self.folder = ROOT / 'results' / (time.strftime('demo-%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
        self.folder.mkdir(parents=True)
        self.path = Path('/dev/shm') / ('yolodemo-'+uuid.uuid4().hex)
        self.bus = InputBus(self.path, create=True)
        self.frames = OrderedDict()
        self.latest = None
        self.jpeg = None
        self.jpeg_sequence = 0
        self.epoch = 0
        self.active = None
        self.desired = args.backend
        self.switching = False
        self.target = None
        self.switch_started = None
        self.switch_ms = None
        self.switches = []
        self.meter = FpsMeter()
        self.workers = {name: dict(state='loading', error=None, info=None, completed=0,
                                  elapsed=0., heartbeat=0., paused=True, pid=None) for name in ('gpu', 'npu')}
        self.processes, self.readers = {}, []
        camera_config = load_config(args.camera_config, args.camera_width, args.camera_height, args.camera_fps)
        self.capture = Capture(args.playlist, args.source, args.camera, self.frame, self.input_state, camera_config)
        base_meta = ROOT / 'models/npu/yolo26n-448.json'
        self.names = json.loads(base_meta.read_text()).get('names', {}) if base_meta.exists() else {}
        self.render_thread = threading.Thread(target=self.render_loop, name='render', daemon=True)
        self.switch_thread = threading.Thread(target=self.switch_loop, name='switch', daemon=True)
        self.initialize_thread = threading.Thread(target=self.initialize, name='initialize', daemon=True)
        self.closed = False

    def start(self):
        self.initialize_thread.start()
        self.render_thread.start()
        self.switch_thread.start()

    def initialize(self):
        # VitisAI output reproducibility can fail during simultaneous HIP
        # initialization on this shared-memory APU. Validate the NPU alone,
        # then initialize HIP; start capture/inference only after both finish.
        for name in ('npu', 'gpu'):
            if self.stop.is_set():
                return
            self.start_worker(name)
            while not self.stop.is_set():
                with self.lock:
                    loading = self.workers[name]['state'] == 'loading'
                if not loading:
                    break
                self.stop.wait(.05)
        if not self.stop.is_set():
            self.capture.thread.start()

    def start_worker(self, name):
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
                   YOLO_CONFIG_DIR=str(ROOT / '.cache/ultralytics'), OPENBLAS_NUM_THREADS='1')
        args = ['-m', 'app.workers.run', '--backend', name, '--bus', str(self.path), '--report-dir', str(self.folder)]
        if name == 'gpu':
            cmd = [str(YOLOFAST / '.venv-rocm10-py314/bin/python'), *args]
        else:
            setup = Path.home() / 'ryzenai_1_8/setup_ryzenai_env.sh'
            # Shell quoting is necessary for paths; source changes only this child.
            script = f'source {shlex.quote(str(setup))} >/dev/null; export PYTHONPATH={shlex.quote(str(ROOT))}; exec python {shlex.join(args)}'
            cmd = ['bash', '-c', 'set -e; '+script]
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            self.processes[name] = proc
            self.workers[name]['pid'] = proc.pid
            reader = threading.Thread(target=self.read_worker, args=(name, proc), name=f'{name}-reader', daemon=True)
            self.readers.append(reader)
            reader.start()
        except Exception as exc:
            self.workers[name].update(state='error', error=str(exc))

    def read_worker(self, name, proc):
        with (self.folder / f'{name}.log').open('w') as log:
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or event.get('backend') != name:
                        raise ValueError('Not a worker event')
                except (ValueError, json.JSONDecodeError):
                    log.write(line)
                    log.flush()
                    continue
                if event['kind'] not in ('result', 'stats'):
                    log.write(line)
                    log.flush()
                self.event(name, event)
        code = proc.wait()
        with self.lock:
            if not self.stop.is_set():
                w = self.workers[name]
                w.update(state='error', error=w['error'] or f'Worker exited ({code})', paused=True)

    def event(self, name, event):
        now = time.monotonic()
        with self.lock:
            w = self.workers[name]
            kind = event['kind']
            if kind == 'ready':
                w.update(state='ready', info=event['info'], error=None, paused=True)
                write_json(self.folder / f'{name}-environment.json', event['info'])
            elif kind == 'state':
                w['state'] = event['state']
            elif kind == 'error':
                w.update(state='error', error=event['error'], paused=True)
                if self.switching and self.target == name:
                    self.switching = False
                    self.active = self.target = None
            elif kind == 'paused':
                w['paused'] = True
                if w['state'] != 'error':
                    w['state'] = 'ready'
            elif kind in ('stats', 'result'):
                w['paused'] = False
                if not self.accept_epoch(name, event['epoch']):
                    return
                w.update(state='active', heartbeat=now, completed=event['completed'], elapsed=event['elapsed'])
                if self.switching:
                    self.active = name
                    self.switching = False
                    self.switch_ms = (now-self.switch_started)*1000
                    self.switches.append(dict(backend=name, epoch=self.epoch, elapsed_ms=self.switch_ms))
                    write_json(self.folder / 'switches.json', self.switches)
                self.meter.update(event['completed'], event['elapsed'])
                if kind == 'result' and event['frame_id'] in self.frames:
                    self.latest = dict(event, received=now)

    def accept_epoch(self, name, epoch):
        return epoch == self.epoch and name == (self.target if self.switching else self.active)

    def frame(self, frame_id, image):
        x, geometry = preprocess(image)
        h, w = image.shape[:2]
        scale = min(1., self.args.display_width/w)
        display = cv2.resize(image, (round(w*scale), round(h*scale))) if scale < 1 else image.copy()
        with self.lock:
            self.frames[frame_id] = (display, geometry)
            while len(self.frames) > 12:
                self.frames.popitem(last=False)
        self.bus.publish(frame_id, x)

    def input_state(self, state):
        self.bus.invalidate()
        with self.lock:
            self.frames.clear()
            self.latest = self.jpeg = None
            w=self.workers.get(self.active)
            self.meter.reset(w['completed'],w['elapsed']) if w else self.meter.reset()
            self.changed.notify_all()

    def request(self, name):
        if name not in self.workers:
            raise ValueError('backend must be gpu or npu')
        with self.lock:
            if self.workers[name]['state'] not in ('ready', 'active'):
                raise ValueError(self.workers[name]['error'] or f'{name.upper()} is not ready')
            self.desired = name

    def switch_loop(self):
        while not self.stop.wait(.01):
            with self.lock:
                desired = self.desired
                w = self.workers[desired]
                if w['state'] == 'error':
                    other = 'npu' if desired == 'gpu' else 'gpu'
                    if self.workers[other]['state'] == 'ready':
                        self.desired = other
                    continue
                if w['state'] not in ('ready', 'active') or self.switching or self.active == desired:
                    continue
                old = self.active
                self.epoch += 1
                self.active = None
                self.target = desired
                self.switching = True
                self.switch_started = time.monotonic()
                self.latest = self.jpeg = None
                self.meter.reset()
                self.bus.select(None, self.epoch)
                self.changed.notify_all()
            # Wait for the old backend's in-flight inference, never overlap load.
            while old and not self.stop.is_set():
                with self.lock:
                    if self.workers[old]['paused']:
                        break
                if time.monotonic()-self.switch_started > 1:
                    proc = self.processes.get(old)
                    if proc and proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                    with self.lock:
                        self.workers[old].update(state='error', error='Pause timed out', paused=True)
                    break
                self.stop.wait(.002)
            self.bus.select(desired, self.epoch)

    def status(self):
        with self.lock:
            now = time.monotonic()
            backend = self.target if self.switching else self.active
            worker = self.workers.get(backend)
            live = worker and now-worker['heartbeat'] < 1 and self.capture.status()['state'] == 'running'
            for w in self.workers.values():
                try:
                    line=next(x for x in Path(f'/proc/{w["pid"]}/status').read_text().splitlines() if x.startswith('VmRSS:'))
                    w['rss_kb']=int(line.split()[1])
                except (OSError,StopIteration,ValueError):w['rss_kb']=None
            data = dict(backend=self.active, desired=self.desired, switching=self.switching,server_pid=os.getpid(),
                epoch=self.epoch, fps=self.meter.fps if live and not self.switching else None,
                input=self.capture.status(), workers={k: v.copy() for k, v in self.workers.items()},
                switch_ms=self.switch_ms, result_frame_id=self.latest['frame_id'] if self.latest else None,
                result_age_ms=(now-self.latest['received'])*1000 if self.latest else None,
                report_dir=str(self.folder), display_width=self.args.display_width, size=448)
            return data

    def render_loop(self):
        previous = None
        while not self.stop.wait(1/30):
            with self.lock:
                event = self.latest
                if not event or time.monotonic()-event['received'] > 1:
                    continue
                key = (event['epoch'], event['frame_id'])
                if key == previous or not self.accept_epoch(event['backend'], event['epoch']):
                    continue
                pair = self.frames.get(event['frame_id'])
                if pair is None:
                    continue
                image, geometry = pair
                image = image.copy()
            detections = restore(event['detections'], geometry)
            sx, sy = image.shape[1]/geometry['width'], image.shape[0]/geometry['height']
            for row in detections:
                cls = int(row[5])
                color = (80+(cls*53)%175, 90+(cls*97)%165, 70+(cls*137)%185)
                x1, y1, x2, y2 = np.rint(row[:4]*[sx,sy,sx,sy]).astype(int)
                cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
                label = f'{self.names.get(str(cls), self.names.get(cls, str(cls)))} {row[4]:.2f}'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)
                ty = max(th+5, y1)
                cv2.rectangle(image, (x1,ty-th-5), (x1+tw+6,ty+2), color, -1)
                cv2.putText(image, label, (x1+3,ty-2), cv2.FONT_HERSHEY_SIMPLEX, .55, (16,18,20), 1, cv2.LINE_AA)
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                continue
            with self.changed:
                if self.latest is None or event['frame_id'] not in self.frames or not self.accept_epoch(event['backend'], event['epoch']):
                    continue
                self.jpeg = encoded.tobytes()
                self.jpeg_sequence += 1
                previous = key
                self.changed.notify_all()

    def next_jpeg(self, sequence):
        with self.changed:
            self.changed.wait_for(lambda: self.stop.is_set() or (self.jpeg is not None and self.jpeg_sequence != sequence), timeout=1)
            return self.jpeg_sequence, self.jpeg

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.initialize_thread.join(timeout=5)
        self.capture.close()
        with self.changed:
            self.changed.notify_all()
        for proc in self.processes.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in self.processes.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for thread in [self.switch_thread, self.render_thread, *self.readers]:
            thread.join(timeout=3)
        write_json(self.folder / 'final-status.json', self.status())
        self.bus.close()
        self.path.unlink(missing_ok=True)
