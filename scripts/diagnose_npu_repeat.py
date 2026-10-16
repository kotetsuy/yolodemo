# SPDX-License-Identifier: AGPL-3.0-only
"""Isolate repeated-inference mismatches without changing the demo or its cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from app.model_utils import ROOT, cases, sha256
from app.preprocess import quantize


def digest(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def difference(a, b):
    ids = np.flatnonzero(a != b)
    if not len(ids):
        return dict(count=0)
    starts = ids[np.r_[True, np.diff(ids) > 1]]
    ends = ids[np.r_[np.diff(ids) > 1, True]]
    return dict(count=int(len(ids)), max_abs=float(np.max(np.abs(a-b))),
                ranges=list(zip(starts.tolist(), ends.tolist()))[:30])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=300)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--head', action='store_true')
    p.add_argument('--reference', action='store_true')
    p.add_argument('--no-arena', action='store_true')
    p.add_argument('--no-pattern', action='store_true')
    p.add_argument('--bind', action='store_true')
    p.add_argument('--retain', action='store_true')
    p.add_argument('--reuse-input', action='store_true')
    p.add_argument('--flush-outputs', action='store_true')
    p.add_argument('--observe-flush', action='store_true')
    p.add_argument('--model', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--opt-level', default='2')
    p.add_argument('--all-inputs', action='store_true')
    p.add_argument('--startup', action='store_true', help='Run the unchanged application startup validator')
    p.add_argument('--production', action='store_true', help='Use the permanent output synchronization shim')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sync = None
    if args.startup or args.production:
        from app.workers.npu_output_sync import bootstrap_output_sync, ensure_output_sync
        bootstrap_output_sync()
        sync = ensure_output_sync()
    if args.startup:
        from app.workers.npu import NPU
        engine = NPU(args.output)
        print(json.dumps(dict(validation_passed=engine.info['validation_passed'],
                              alternating_inputs_passed=engine.info['alternating_inputs_passed'])), flush=True)
        engine.close()
        return
    import onnxruntime as ort
    selected = json.loads((ROOT/'models/npu/selection.json').read_text())
    model = args.model or ROOT/selected['model']
    meta = json.loads((ROOT/selected['model']).with_suffix('.json').read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.enable_cpu_mem_arena = not args.no_arena
    options.enable_mem_pattern = not args.no_pattern
    cache = args.cache or ROOT/'.cache/npu-448'/sha256(model)/f'opt{args.opt_level}'
    providers = ['CPUExecutionProvider'] if args.cpu else ['VitisAIExecutionProvider', 'CPUExecutionProvider']
    provider_options = [{}] if args.cpu else [dict(cache_dir=str(cache), opt_level=args.opt_level, enable_cache_file_io_in_mem='0'), {}]
    session = ort.InferenceSession(str(model), options, providers=providers, provider_options=provider_options)
    session.disable_fallback()
    if sync:
        sync.verify_loaded()
    if not args.cpu and 'VitisAIExecutionProvider' not in session.get_providers():
        raise RuntimeError('NPU provider unavailable')
    head = ort.InferenceSession(meta['cpu_head_model'], options, providers=['CPUExecutionProvider']) if args.head else None
    ref = ort.InferenceSession(meta['reference_model'], options, providers=['CPUExecutionProvider']) if args.reference else None
    inputs = [x for _, x in cases()]
    quantized = [quantize(x, meta['scale'], meta['zero_point'], np.empty_like(x), np.empty(x.shape, np.int16)) for x in inputs]
    hashes = [digest(x) for x in quantized]
    # Keep all inputs and baseline outputs alive, independently allocated.
    for x in quantized:
        x.flags.writeable = False
    name = session.get_inputs()[0].name
    q = np.empty_like(quantized[0])
    scratch = np.empty_like(inputs[0])
    bound = None
    if args.bind:
        bound = session.io_binding()
        buffers = [np.empty(o.shape, dtype=np.float32) for o in session.get_outputs()]
        for node, buf in zip(session.get_outputs(), buffers):
            bound.bind_output(node.name, 'cpu', 0, buf.dtype, buf.shape, buf.ctypes.data)
    saved, history, events = {}, {}, []
    flush = None
    flushed_bytes = 0
    if args.flush_outputs or args.observe_flush:
        import ctypes
        flush = ctypes.CDLL(None).diagnostic_flush_outputs
        flush.argtypes = [ctypes.c_int]
        flush.restype = ctypes.c_size_t
    baseline_raw = []
    started = time.monotonic()
    pattern = list(range(8)) if args.all_inputs else [0, 1, 2, 0, 2, 1, 0, 1, 2, 2, 1, 0]
    order = list(range(8)) + [pattern[i % len(pattern)] for i in range(args.iterations)]
    for step, i in enumerate(order):
        feed = quantize(inputs[i], meta['scale'], meta['zero_point'], scratch, q) if args.reuse_input else quantized[i]
        if bound is None:
            raw = sync.run(session, {name: feed}) if sync else session.run(None, {name: feed})
        else:
            bound.bind_cpu_input(name, feed)
            session.run_with_iobinding(bound)
            raw = buffers
        copies = [v.copy() for v in raw]
        # Observe output before CPU head/reference execution and check for writes.
        if head:
            head.run(None, dict(zip(meta['cpu_head_input_names'], raw[1:])))
        if ref:
            ref.run(None, {'images': inputs[i]})
        mutation = [difference(a, b) for a, b in zip(copies, raw)]
        retained_mutation = [] if bound is not None else [difference(a, b) for old, snap in baseline_raw for a, b in zip(old, snap)]
        if i not in saved:
            saved[i] = copies
            if bound is None and args.retain:
                baseline_raw.append((raw, copies))
        diffs = [difference(a, b) for a, b in zip(saved[i], copies)]
        if any(d['count'] for d in diffs + mutation + retained_mutation):
            event = dict(step=step, input=i, difference=diffs, mutated_by_cpu=mutation,
                         retained_changed=sum(d['count'] for d in retained_mutation))
            # Test whether bad regions exactly match an earlier, different input.
            for j, (a, b) in enumerate(zip(saved[i], copies)):
                mask = a != b
                if mask.any():
                    event.setdefault('previous_input_matches', {})[str(j)] = {
                        str(k): int(np.count_nonzero(b[mask] == v[j][mask])) for k, v in history.items()}
            events.append(event)
            if len(events) <= 5:
                np.savez(args.output/f'mismatch-{step}.npz', input=quantized[i],
                         **{f'expected_{j}':v for j,v in enumerate(saved[i])},
                         **{f'actual_{j}':v for j,v in enumerate(copies)})
                print(json.dumps(event), flush=True)
        history[i] = copies
        if flush:
            flushed_bytes = flush(int(args.flush_outputs))
        if step and step % 100 == 0:
            print(json.dumps(dict(progress=step, mismatches=len(events))), flush=True)
    np.savez(args.output/'baseline.npz', **{f'input_{i}_output_{j}':v for i, arrays in saved.items() for j,v in enumerate(arrays)})
    report = dict(arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  model=str(model), model_sha256=sha256(model), ort=ort.__version__, python=platform.python_version(),
                  providers=session.get_providers(), outputs=[o.name for o in session.get_outputs()],
                  elapsed=time.monotonic()-started, runs=len(order), mismatch_runs=len(events), events=events,
                  inputs_unchanged=hashes == [digest(x) for x in quantized],
                  output_bo_bytes=flushed_bytes,
                  environment={k:v for k,v in os.environ.items() if k.startswith(('XLNX','VAIP','XRT','RYZEN','ORT','OMP','OPENBLAS','NUM_OF_','DEMO_DIAG','LD_PRELOAD'))})
    if flush:
        for key in ('diagnostic_pdi_calls', 'diagnostic_post_sync_calls'):
            try:
                counter = getattr(ctypes.CDLL(None), key)
                counter.restype = ctypes.c_size_t
                report[key] = counter()
            except AttributeError:
                pass
    if sync:
        report['output_sync'] = sync.info()
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('runs','mismatch_runs','inputs_unchanged','elapsed')}), flush=True)


if __name__ == '__main__':
    main()
