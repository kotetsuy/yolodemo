# SPDX-License-Identifier: AGPL-3.0-only
"""Continuous inference; stdout is a bounded-rate JSON event stream."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback
from app.ipc import InputBus, BACKENDS


def emit(**data):
    print(json.dumps(data, separators=(',', ':'), allow_nan=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['gpu', 'npu'], required=True)
    p.add_argument('--bus', required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    args = p.parse_args()
    if args.backend == 'npu':
        try:
            from app.workers.npu_output_sync import bootstrap_output_sync
            bootstrap_output_sync()
        except Exception as exc:
            emit(kind='error', state='error', backend='npu', error=str(exc))
            traceback.print_exc()
            return 1
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    bus = InputBus(args.bus)
    engine = None
    try:
        emit(kind='state', state='loading', backend=args.backend)
        if args.backend == 'gpu':
            from app.workers.gpu import GPU
            engine = GPU(args.report_dir)
        else:
            from app.workers.npu import NPU
            engine = NPU(args.report_dir)
        emit(kind='ready', state='ready', backend=args.backend, info=engine.info)
        frame_id, epoch, completed = -1, -1, 0
        started = last_stats = time.monotonic()
        last_result = 0.
        active = False
        while not stopping:
            control, x = bus.read(frame_id, for_backend=BACKENDS[args.backend])
            wanted = control['selected'] == BACKENDS[args.backend] and control['valid'] and time.monotonic() - control['stamp'] < 2
            if not wanted:
                if active:
                    emit(kind='paused', backend=args.backend, epoch=epoch, completed=completed, elapsed=time.monotonic()-started)
                    active = False
                time.sleep(.005)
                continue
            if control['epoch'] != epoch:
                epoch = control['epoch']
                completed = 0
                started = last_stats = time.monotonic()
                active = False
            if not active:
                if x is None:
                    control, x = bus.read(-1, for_backend=BACKENDS[args.backend])
                    if x is None or control['epoch'] != epoch or not control['valid']:
                        continue
                active = True
                frame_id = -1
                last_result = 0.
            fresh = x is not None
            if fresh:
                frame_id = control['frame_id']
                engine.prepare(x)
            engine.infer()
            completed += 1
            now = time.monotonic()
            if fresh and now - last_result >= 1/30:
                detections = engine.detections()
                if not __import__('numpy').isfinite(detections).all():
                    raise RuntimeError('Nonfinite detection output')
                detections = detections[detections[:,4] >= .25]
                emit(kind='result', backend=args.backend, epoch=epoch, frame_id=frame_id,
                     detections=detections.tolist(), completed=completed, elapsed=now-started)
                last_result = now
            if now - last_stats >= .3:
                emit(kind='stats', backend=args.backend, epoch=epoch,
                     completed=completed, elapsed=now-started)
                last_stats = now
    except Exception as exc:
        emit(kind='error', state='error', backend=args.backend, error=str(exc))
        traceback.print_exc()
        return 1
    finally:
        if engine is not None:
            engine.close()
        bus.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
