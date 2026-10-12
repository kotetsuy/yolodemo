# SPDX-License-Identifier: AGPL-3.0-only
import json
import platform
import os
from pathlib import Path
import numpy as np
from app.model_utils import MODEL, ROOT, sha256, cases, decode, compare_detections, write_json, output_pair
from app.preprocess import SIZE, quantize


class NPU:
    boundary = 'cached normalized CPU FP32 input -> external INT16 quantization -> ORT session.run -> CPU stable Top-k'

    def __init__(self, report_dir):
        import onnxruntime as ort
        model = Path(os.environ.get('NPU_MODEL', str(MODEL)))
        selection = ROOT / 'models/npu/selection.json'
        if 'NPU_MODEL' not in os.environ and selection.exists():
            selected = json.loads(selection.read_text())
            model = ROOT / selected['model']
            if selected['sha256'] != sha256(model):
                raise RuntimeError('Selected NPU model hash changed')
        if not list(Path('/dev/accel').glob('accel*')) or 'VitisAIExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('NPU device or VitisAI EP is unavailable')
        self.meta = json.loads(model.with_suffix('.json').read_text())
        if self.meta.get('status') != 'complete' or self.meta['model_sha256'] != sha256(model):
            raise RuntimeError('NPU model provenance is invalid; run scripts/prepare_models.sh')
        cache = ROOT / '.cache/npu-448' / sha256(model)
        cache.mkdir(parents=True, exist_ok=True)
        level = os.environ.get('NPU_OPT_LEVEL', '2')
        cache = cache / f'opt{level}'
        cache.mkdir(parents=True, exist_ok=True)
        opts = dict(cache_dir=str(cache), opt_level=level, enable_cache_file_io_in_mem='0')
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model), so,
            providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'], provider_options=[opts, {}])
        self.session.disable_fallback()
        if 'VitisAIExecutionProvider' not in self.session.get_providers():
            raise RuntimeError('NPU session fell back to CPU')
        node = self.session.get_inputs()[0]
        if node.shape != [1, 3, SIZE, SIZE] or node.type != 'tensor(int16)':
            raise RuntimeError(f'Wrong NPU input: {node.shape} {node.type}')
        expected_shapes = self.meta['output_shapes']
        if [o.shape for o in self.session.get_outputs()] != expected_shapes:
            raise RuntimeError('Wrong NPU output shapes')
        self.input_name = node.name
        self.head=None
        if self.meta.get('cpu_head_model'):
            if sha256(self.meta['cpu_head_model'])!=self.meta['cpu_head_sha256']:
                raise RuntimeError('CPU classifier model changed')
            head_opts=ort.SessionOptions()
            head_opts.intra_op_num_threads=4
            self.head=ort.InferenceSession(self.meta['cpu_head_model'],head_opts,providers=['CPUExecutionProvider'])
            self.boundary = 'cached normalized CPU FP32 input -> external INT16 quantization -> NPU backbone/boxes -> CPU A16W8 classifier with preceding SiLU -> stable Top-k'
        self.scratch = np.empty(node.shape, np.float32)
        self.q = np.empty(node.shape, np.int16)
        self.info = dict(backend='npu', python=platform.python_version(), ort=ort.__version__,
            shape=node.shape, precision='a16w8-perchannel', model_sha256=sha256(model), model=str(model),
            cpu_math_nodes=self.meta.get('cpu_math_nodes', []),
            boundary=self.boundary, provider_options=opts, validation=[])
        self.info['variant']=self.meta.get('variant','baseline')
        self.info['cpu_classifier']=bool(self.head)
        refpath = Path(self.meta['reference_model'])
        if sha256(refpath) != self.meta['reference_sha256']:
            raise RuntimeError('NPU reference model changed')
        cpu_opts = ort.SessionOptions()
        cpu_opts.intra_op_num_threads = 4
        reference = ort.InferenceSession(str(refpath), cpu_opts, providers=['CPUExecutionProvider'])
        saved = []
        try:
            for record, x in cases():
                self.prepare(x)
                self.infer()
                expected = reference.run(None, {'images': x})
                check = compare_detections(decode(*output_pair(expected, self.meta)), self.output)
                self.info['validation'].append(dict(input=record, **check))
                saved.append((x, [v.copy() for v in self.raw], self.output.copy()))
                if not check['passed']:
                    np.savez(report_dir / 'npu-mismatch.npz', input=x, boxes=self.raw[0], scores=self.raw[1], detections=self.output)
                    raise RuntimeError('448 CPU/NPU smoke validation failed')
            for i in [0, 1, 2, 0, 2, 1, 0, 1, 2, 2, 1, 0]:
                x, raw, out = saved[i]
                self.prepare(x)
                self.infer()
                if not all(np.array_equal(a, b) for a, b in zip(raw, self.raw)) or not np.array_equal(out, self.output):
                    self.info['alternating_mismatch'] = dict(input_index=i,
                        raw_max_abs=[float(np.max(np.abs(a-b))) for a,b in zip(raw,self.raw)],
                        detection_check=compare_detections(out,self.output))
                    np.savez(report_dir/'npu-alternating-mismatch.npz',input=x,
                        expected_detections=out,actual_detections=self.output,
                        **{f'expected_raw_{j}':v for j,v in enumerate(raw)},
                        **{f'actual_raw_{j}':v for j,v in enumerate(self.raw)})
                    raise RuntimeError('NPU alternating input equivalence failed')
            for _ in range(100):
                self.infer()
            self.info['validation_passed'] = True
            self.info['alternating_inputs_passed'] = True
            write_json(report_dir / 'npu-validation.json', self.info)
        except Exception as exc:
            write_json(report_dir / 'npu-validation.json', dict(self.info, validation_passed=False, error=str(exc)))
            raise

    def prepare(self, x):
        self.x = x

    def infer(self):
        quantize(self.x, self.meta['scale'], self.meta['zero_point'], self.scratch, self.q)
        self.raw = self.session.run(None, {self.input_name: self.q})
        if self.head:
            scores=self.head.run(None,dict(zip(self.meta['cpu_head_input_names'],self.raw[1:])))[0]
            self.output=decode(self.raw[0],scores)
        else:
            self.output = decode(*output_pair(self.raw, self.meta))

    def detections(self):
        return self.output.copy()

    def close(self):
        pass
