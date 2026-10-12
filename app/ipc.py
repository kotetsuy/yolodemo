# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded cross-interpreter shared memory, with flock and immutable copies.

An mmap file in /dev/shm avoids Python 3.12/3.14 resource-tracker ownership
differences. Only the coordinator owns/unlinks it; workers attach independently.
"""
import fcntl
import mmap
from pathlib import Path
import struct
import threading
import time
import numpy as np
from app.preprocess import SIZE

HEADER = struct.Struct('<QQQQd')
OFFSET = 4096
SHAPE = (1, 3, SIZE, SIZE)
LENGTH = OFFSET + int(np.prod(SHAPE)) * 4
BACKENDS = {None: 0, 'gpu': 1, 'npu': 2}


class InputBus:
    def __init__(self, path, create=False):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.file = self.path.open('w+b' if create else 'r+b')
        if create:
            self.file.truncate(LENGTH)
        self.mem = mmap.mmap(self.file.fileno(), LENGTH)
        self.array = np.ndarray(SHAPE, dtype=np.float32, buffer=self.mem, offset=OFFSET)
        if create:
            self.mem[:HEADER.size] = HEADER.pack(0, 0, 0, 0, 0.)

    def _enter(self):
        self.lock.acquire()
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)

    def _exit(self):
        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.lock.release()

    def publish(self, frame_id, x):
        self._enter()
        try:
            selected, epoch, _, _, _ = HEADER.unpack(self.mem[:HEADER.size])
            np.copyto(self.array, x)
            self.mem[:HEADER.size] = HEADER.pack(selected, epoch, frame_id, 1, time.monotonic())
        finally:
            self._exit()

    def select(self, backend, epoch):
        self._enter()
        try:
            _, _, frame_id, valid, stamp = HEADER.unpack(self.mem[:HEADER.size])
            self.mem[:HEADER.size] = HEADER.pack(BACKENDS[backend], epoch, frame_id, valid, stamp)
        finally:
            self._exit()

    def invalidate(self):
        self._enter()
        try:
            selected, epoch, frame_id, _, stamp = HEADER.unpack(self.mem[:HEADER.size])
            self.mem[:HEADER.size] = HEADER.pack(selected, epoch, frame_id, 0, stamp)
        finally:
            self._exit()

    def read(self, previous_id=-1, for_backend=None):
        self._enter()
        try:
            selected, epoch, frame_id, valid, stamp = HEADER.unpack(self.mem[:HEADER.size])
            fresh = valid and frame_id != previous_id and (for_backend is None or selected == for_backend)
            x = self.array.copy() if fresh else None
            return dict(selected=selected, epoch=epoch, frame_id=frame_id,
                        valid=bool(valid), stamp=stamp), x
        finally:
            self._exit()

    def close(self):
        del self.array
        self.mem.close()
        self.file.close()
