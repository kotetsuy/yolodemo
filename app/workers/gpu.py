# SPDX-License-Identifier: AGPL-3.0-only
"""FP32 MIGraphX worker following yolofast's validated 448 inference path."""
import os
import platform
from app.model_utils import WEIGHTS, YOLOFAST, ROOT, cases, sha256, write_json, compare_detections
from app.preprocess import SIZE


class GPU:
    boundary = 'cached GPU FP32 input -> MIGraphX HIP graph -> synchronized GPU detections; live loop includes input updates and sampled output copies'

    def __init__(self, report_dir):
        os.environ['ORT_MIGRAPHX_HIP_GRAPH_ENABLE'] = '1'
        os.environ.setdefault('ORT_MIGRAPHX_CACHE_DIR', str(ROOT / '.cache/migraphx'))
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        import torch
        import onnxruntime as ort
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.nn.backends.onnx import ONNXBackend
        self.torch = torch
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('ROCm GPU is unavailable')
        torch.set_num_threads(4)
        model = YOLOFAST / 'models/migraphx/yolo26n-448-fp32.onnx'
        if not model.is_file():
            raise RuntimeError(f'MIGraphX model missing: {model}; run yolofast scripts/export_migraphx.sh')
        self.backend = ONNXBackend(str(model), device=torch.device('cuda:0'))
        providers = self.backend.session.get_providers()
        if self.backend.device.type != 'cuda' or not self.backend.use_io_binding or providers[0] != 'MIGraphXExecutionProvider':
            raise RuntimeError('MIGraphX GPU with IO binding required; CPU fallback rejected')
        inputs = self.backend.session.get_inputs()
        outputs = self.backend.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, SIZE, SIZE] or inputs[0].type != 'tensor(float)' or len(outputs) != 1 or outputs[0].shape != [1, 300, 6]:
            raise RuntimeError('Expected fixed FP32 448 input and 300x6 end2end detections')
        self.x = torch.zeros((1, 3, SIZE, SIZE), device='cuda', dtype=torch.float32)
        self.host = torch.empty(self.x.shape, dtype=torch.float32, pin_memory=True)
        self.input_value = ort.OrtValue.from_dlpack(self.x)
        self.backend.io.bind_ortvalue_input(inputs[0].name, self.input_value)
        self.output = self.backend.bindings[0]
        self.names = self.backend.names
        self.info = dict(backend='gpu', engine='migraphx', device=torch.cuda.get_device_name(0),
                         python=platform.python_version(), torch=torch.__version__, hip=torch.version.hip,
                         ultralytics=ultralytics.__version__, onnxruntime=ort.__version__, providers=providers,
                         shape=list(self.x.shape), precision='fp32', model_sha256=sha256(model),
                         weights_sha256=sha256(WEIGHTS), hip_graph=True, prebind=True,
                         boundary=self.boundary, validation_criteria=dict(confidence=.25, min_iou=.95, max_score_diff=.03),
                         validation=[])
        try:
            with torch.inference_mode():
                reference = YOLO(str(WEIGHTS)).model.eval().cuda().float()
                reference.end2end = True
                reference.fuse(verbose=False)
                for record, x in cases():
                    self.prepare(x)
                    expected = reference(self.x)[0][0].cpu().numpy()
                    self.infer()
                    actual = self.detections()
                    comparison = compare_detections(expected, actual, min_iou=.95, max_score_diff=.03)
                    self.info['validation'].append(dict(input=record, **comparison))
                    if not comparison['passed']:
                        raise RuntimeError('FP32 PyTorch/MIGraphX detection comparison failed')
                del reference
                for _ in range(100):
                    self.infer()
            self.info['validation_passed'] = True
            write_json(report_dir / 'gpu-validation.json', self.info)
        except Exception as exc:
            write_json(report_dir / 'gpu-validation.json', dict(self.info, validation_passed=False, error=str(exc)))
            raise

    def prepare(self, x):
        self.host.copy_(self.torch.from_numpy(x))
        self.x.copy_(self.host, non_blocking=True)
        # ORT and PyTorch can use different HIP streams.
        self.torch.cuda.synchronize()

    def infer(self):
        self.backend.session.run_with_iobinding(self.backend.io)
        self.torch.cuda.synchronize()

    def detections(self):
        return self.output[0].float().cpu().numpy().copy()

    def close(self):
        self.torch.cuda.synchronize()
