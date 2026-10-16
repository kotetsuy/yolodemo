# SPDX-License-Identifier: AGPL-3.0-only
"""Hardware regression: exact alternate-input outputs across session lifetimes."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.model_utils import cases, write_json
from app.workers.npu_output_sync import bootstrap_output_sync, ensure_output_sync


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/npu-permanent-fix/regression'))
    parser.add_argument('--sessions', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=2000)
    args = parser.parse_args()
    if args.sessions < 1 or args.iterations < 1:
        parser.error('Positive sessions and iterations required')
    bootstrap_output_sync()
    import numpy as np
    from app.workers.npu import NPU
    guard = ensure_output_sync()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = [x for _, x in cases()]
    baseline = None
    report = dict(passed=False, sessions=[], iterations=args.iterations)
    started = time.monotonic()
    try:
        for trial in range(args.sessions):
            folder = args.output/f'session-{trial+1}'
            folder.mkdir(exist_ok=True)
            engine = NPU(folder)
            try:
                if baseline is None:
                    baseline = []
                    for x in inputs:
                        engine.prepare(x)
                        engine.infer()
                        baseline.append(([v.copy() for v in engine.raw], engine.output.copy()))
                for step in range(args.iterations):
                    index = step % len(inputs)
                    engine.prepare(inputs[index])
                    engine.infer()
                    raw, detections = baseline[index]
                    if (not all(np.array_equal(a, b) for a, b in zip(raw, engine.raw))
                            or not np.array_equal(detections, engine.output)):
                        raise RuntimeError(f'Output mismatch: session {trial+1}, step {step}, input {index}')
            finally:
                engine.close()
            stats = guard.stats()
            if stats['live_buffers'] or stats['npu_completions'] != stats['output_syncs']:
                raise RuntimeError('Native buffers or unsynchronized completions remain after session close')
            report['sessions'].append(dict(session=trial+1, validation_passed=True,
                                           exact_repeats=args.iterations, **stats))
            print(report['sessions'][-1], flush=True)
        report['passed'] = True
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        report.update(output_sync=guard.info(), elapsed=time.monotonic()-started)
        write_json(args.output/'report.json', report)


if __name__ == '__main__':
    main()
