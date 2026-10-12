# SPDX-License-Identifier: AGPL-3.0-only
"""Hotplug-aware camera selection and real-time local playlist acquisition."""
from collections import deque
import multiprocessing
from pathlib import Path
import queue
import threading
import time
import cv2
from app.cameras import enumerate_cameras, camera_inventory, candidates, rank, label, matches, load_config, configure_format


def playlist_entries(path):
    path = Path(path).expanduser().resolve()
    entries = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            entries.append((path.parent / line).resolve())
    if not entries:
        raise ValueError('Empty playlist')
    return entries


def _latest_put(output, packet):
    try:
        output.put_nowait(packet)
    except queue.Full:
        try:
            output.get_nowait()
        except queue.Empty:
            pass
        try:
            output.put_nowait(packet)
        except queue.Full:
            pass


def camera_producer(device, requested, output, stop):
    cv2.setNumThreads(1)
    cap = None
    try:
        for fourcc in requested['fourcc_priority']:
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            fmt = configure_format(cap, fourcc, requested)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue
            failures = 0
            while not stop.is_set():
                fmt['height'], fmt['width'] = frame.shape[:2]
                _latest_put(output, (frame, fmt.copy(), time.monotonic(), None))
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 3:
                        _latest_put(output, (None, None, time.monotonic(), 'Camera read failed'))
                        return
                    time.sleep(.05)
                    # Retry reads, without publishing a missing frame.
                    while not stop.is_set() and (not ok or frame is None) and failures < 3:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            failures += 1
                            time.sleep(.05)
                    if not ok or frame is None:
                        _latest_put(output, (None, None, time.monotonic(), 'Camera read failed'))
                        return
                failures = 0
            return
        _latest_put(output, (None, None, time.monotonic(), 'Camera could not be opened'))
    except Exception as exc:
        _latest_put(output, (None, None, time.monotonic(), str(exc)))
    finally:
        if cap is not None:
            cap.release()
        output.cancel_join_thread()


class CameraSession:
    """A single bounded latest-frame queue and a killable V4L2 process."""
    def __init__(self, device, requested):
        self.device = device
        ctx = multiprocessing.get_context('spawn')
        self.output, self.stopped = ctx.Queue(maxsize=1), ctx.Event()
        self.proc = ctx.Process(target=camera_producer,
            args=(device.path, requested, self.output, self.stopped), name='camera')
        self.last_frame = self.started = time.monotonic()
        self.received = False
        self.proc.start()

    def poll(self):
        try:
            packet = self.output.get_nowait()
        except queue.Empty:
            if not self.proc.is_alive():
                return (None, None, time.monotonic(), 'Camera process exited')
            timeout = 2 if self.received else 5
            if time.monotonic()-self.last_frame > timeout:
                return (None, None, time.monotonic(), 'Camera timed out')
            return None
        if packet[0] is not None:
            self.last_frame = packet[2]
            self.received = True
        return packet

    def close(self):
        self.stopped.set()
        self.proc.join(timeout=.2)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=1)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=1)
        self.output.close()
        self.output.cancel_join_thread()
        if not self.proc.is_alive():
            self.proc.close()


class PlaylistPlayer:
    def __init__(self, path):
        self.entries = playlist_entries(path)
        self.index = self.failed = self.read_count = 0
        self.cap = None
        self.name = None
        self.deadline = 0.

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.name = None

    def read(self):
        if self.cap is None:
            path = self.entries[self.index]
            self.index = (self.index+1) % len(self.entries)
            cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 8]) if path.is_file() else None
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                self.failed += 1
                if self.failed >= len(self.entries):
                    raise RuntimeError('All playlist videos are unreadable')
                return None
            self.cap, self.name, self.read_count = cap, path.name, 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.period = 1/(fps if 1 <= fps <= 240 else 30.)
            self.deadline = time.monotonic()
        now = time.monotonic()
        if now < self.deadline:
            return None
        ok, frame = self.cap.read()
        if not ok:
            self.failed = 0 if self.read_count else self.failed+1
            self.close()
            if self.failed >= len(self.entries):
                raise RuntimeError('All playlist videos are empty')
            return None
        if self.read_count == 0:
            self.deadline = time.monotonic()
        self.read_count += 1
        late = time.monotonic()-self.deadline
        self.deadline += self.period
        return frame if late < 2*self.period else None


class Capture:
    def __init__(self, playlist, source, camera, on_frame, on_state, config=None):
        self.playlist, self.source, self.camera = playlist, source, camera
        self.config = config if config is not None else load_config()
        self.selection = 'playlist' if source == 'playlist' else str(camera) if camera is not None else 'auto'
        self.revision = 0
        self.devices = []
        self.on_frame, self.on_state = on_frame, on_state
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name='capture', daemon=True)
        self.frame_id = self.total = 0
        self.times = deque()
        self.state = dict(state='loading', source=None, fps=0., total_frames=0, reason=None,
                          camera=None, cameras=[], width=None, height=None, pending_camera=None, generation=0)
        self.lock = threading.Lock()

    def status(self):
        with self.lock:
            data = self.state.copy()
            data['camera_selection'] = self.selection
            data['requested_format'] = self.config['format'].copy()
            data['fps'] = data['fps'] if self.times and time.monotonic()-self.times[-1] < 1 else 0.
            return data

    def request_camera(self, selection):
        if not isinstance(selection, str):
            raise ValueError('camera must be a string')
        with self.lock:
            if selection not in ('auto', 'playlist'):
                chosen = next((d for d in self.devices if matches(d, selection)), None)
                if chosen is None:
                    raise ValueError('選択したカメラは接続されていません')
                selection = chosen.id
            if selection != self.selection:
                self.selection = selection
                self.revision += 1

    def change(self, state, name, reason=None):
        with self.lock:
            self.state.update(state=state, source=name, reason=reason)
            if state != 'running':
                self.state['generation'] += 1
                self.times.clear()
                self.state['fps'] = 0.
        if state != 'running':
            self.on_state(state)

    def deliver(self, frame, name):
        now = time.monotonic()
        self.frame_id += 1
        self.total += 1
        with self.lock:
            self.times.append(now)
            while len(self.times) > 1 and now-self.times[0] > 2:
                self.times.popleft()
            fps = (len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times)>1 else 0.
            self.state.update(state='running', source=name, fps=fps, total_frames=self.total,
                              width=frame.shape[1], height=frame.shape[0])
        self.on_frame(self.frame_id, frame)

    def run(self):
        active = pending = player = None
        failed, seen = {}, {}
        next_scan, previous_revision = 0., -1
        reason = None
        try:
            while not self.stop.is_set():
                now = time.monotonic()
                if now >= next_scan:
                    devices = enumerate_cameras()
                    with self.lock:
                        self.devices = devices
                        self.state['cameras'] = camera_inventory(devices, self.config)
                    present = {d.id for d in devices}
                    seen = {key:value for key,value in seen.items() if key in present}
                    for device in devices:
                        seen.setdefault(device.id, now)
                    next_scan = now+1
                with self.lock:
                    selection, revision = self.selection, self.revision
                    devices = self.devices.copy()
                if revision != previous_revision:
                    if pending is not None:
                        pending.close();pending = None
                    previous_revision = revision
                if active is not None and not any(d.id == active.device.id and d.path == active.device.path for d in devices):
                    active.close();active = None
                    reason = 'カメラが切断されたため別の入力へ移行'
                    with self.lock:self.state['camera'] = None
                    self.change('loading', None, reason)
                ordered = [] if selection == 'playlist' else candidates(devices, self.config, selection)
                ordered = [d for d in ordered if now >= failed.get(d.id, 0.) or (active and d.id == active.device.id)]
                desired = ordered[0] if ordered else None
                if active is not None and desired is not None and desired.id != active.device.id:
                    manual = selection != 'auto' and matches(desired, selection)
                    if not manual and (not self.config['preempt'] or
                        rank(desired, self.config) >= rank(active.device, self.config) or
                        now-seen[desired.id] < float(self.config['preempt_settle_sec'])):
                        desired = active.device
                if pending is not None and (desired is None or pending.device.id != desired.id or pending.device.path != desired.path):
                    pending.close();pending = None
                if selection == 'playlist' and active is not None:
                    active.close();active = None
                    with self.lock:self.state['camera'] = None
                    self.change('loading', None, '動画を選択しました')
                if desired is not None and pending is None and (active is None or desired.id != active.device.id):
                    pending = CameraSession(desired, self.config['format'])
                with self.lock:
                    self.state['pending_camera'] = pending.device.id if pending else None
                first = None
                if pending is not None:
                    packet = pending.poll()
                    if packet is not None:
                        if packet[0] is None:
                            failed[pending.device.id] = now+5
                            reason = f'{label(pending.device,self.config)}: {packet[3]}'
                            pending.close();pending = None
                        else:
                            if active is not None:
                                active.close()
                            active, pending, first = pending, None, packet
                            if player is not None:player.close()
                            self.change('loading', label(active.device,self.config))
                if active is not None:
                    packet = first or active.poll()
                    if packet is not None:
                        frame, fmt, stamp, error = packet
                        if frame is None:
                            failed[active.device.id] = now+5
                            reason = f'{label(active.device,self.config)}: {error}'
                            active.close();active = None
                            with self.lock:self.state['camera'] = None
                            self.change('loading', None, reason)
                        else:
                            with self.lock:
                                self.state['camera'] = dict(id=active.device.id, path=active.device.path,
                                    label=label(active.device,self.config), **fmt)
                            self.deliver(frame, label(active.device,self.config))
                if active is None:
                    if player is None:player = PlaylistPlayer(self.playlist)
                    frame = player.read()
                    if frame is not None:
                        if self.state['source'] != player.name or self.state['state'] != 'running':
                            reason = ('動画を選択しました' if selection == 'playlist' else reason or
                                      'カメラが見つからないため動画を使用')
                            self.change('loading', player.name, reason)
                        self.deliver(frame, player.name)
                self.stop.wait(.001 if active is None else .005)
        except Exception as exc:
            self.change('error', None, str(exc))
        finally:
            for session in (pending, active):
                if session is not None:session.close()
            if player is not None:player.close()

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
