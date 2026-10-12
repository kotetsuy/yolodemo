# SPDX-License-Identifier: AGPL-3.0-only
import os
import platform
import time
import numpy as np
from app.model_utils import WEIGHTS, ROOT, cases, sha256, write_json, decode
from app.preprocess import SIZE


class GPU:
    boundary = 'cached GPU FP16 input -> HIP graph -> synchronized GPU detections; live loop includes input updates and sampled output copies'

    def __init__(self, report_dir):
        import torch
        import ultralytics
        from ultralytics import YOLO
        self.torch = torch
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('ROCm GPU is unavailable')
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        self.model = YOLO(str(WEIGHTS)).model.eval().to('cuda').float()
        self.model.end2end = True
        self.model.fuse(verbose=False)
        self.model.half()
        # Equal-score Top-k picks can differ between eager and captured kernels.
        # Preserve the two-stage selection, using index order to break ties on
        # GPU, exactly like the project's stable CPU decoder.
        def stable_topk(scores, k, groups=1):
            if groups != 1:
                raise RuntimeError('Only ungrouped batch=1 Top-k is supported')
            # Positive FP16 probabilities have monotonic INT16 bit patterns.
            # A unique INT32 lexicographic key retains every original score
            # bit while breaking ties by index; no epsilon perturbs scores.
            if scores.dtype != torch.float16 or scores.shape[-1] >= 32768:
                raise RuntimeError('Stable Top-k key requires FP16 and <32768 entries')
            ranks = scores.view(torch.int16).to(torch.int32)
            keys = (ranks << 15) - torch.arange(scores.shape[-1], device=scores.device, dtype=torch.int32)
            index = keys.topk(k, dim=-1, sorted=True).indices
            return scores.gather(-1, index), index
        self.model.model[-1]._grouped_topk = stable_topk
        self.names = self.model.names
        self.x = torch.zeros((1, 3, SIZE, SIZE), device='cuda', dtype=torch.float16)
        self.host = torch.empty(self.x.shape, dtype=torch.float16, pin_memory=True)
        self.graph = torch.cuda.CUDAGraph()
        self.context = torch.inference_mode()
        self.context.__enter__()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(100):
                self.model(self.x)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, stream=stream):
            self.captured = self.model(self.x)
        self.output = self.captured[0]
        self.done = torch.cuda.Event()
        self.info = dict(backend='gpu', device=torch.cuda.get_device_name(0), python=platform.python_version(),
                         torch=torch.__version__, hip=torch.version.hip, ultralytics=ultralytics.__version__,
                         shape=list(self.x.shape), precision='fp16', model_sha256=sha256(WEIGHTS),
                         boundary=self.boundary, validation=[])
        self.info['topk'] = 'stable two-stage INT32 lexicographic Top-k; FP16 scores unchanged'
        try:
            for record, x in cases():
                self.prepare(x)
                expected = self.model(self.x)[0].clone()
                head=self.model.model[-1]
                postprocess=head.postprocess
                try:
                    head.postprocess=lambda preds:preds
                    raw=self.model(self.x)[0].cpu().numpy()
                finally:
                    head.postprocess=postprocess
                cpu_expected=decode(raw[:,:,:4].transpose(0,2,1),raw[:,:,4:].transpose(0,2,1))
                decoder_exact=np.array_equal(expected[0].cpu().numpy(),cpu_expected)
                self.infer()
                finite = bool(expected.isfinite().all() & self.output.isfinite().all())
                ordered_exact = torch.equal(expected, self.output)
                # Rendering treats detections as a multiset. Top-k can reorder
                # equal scores; require bit-exact rows including multiplicities.
                a, b = expected[0].cpu().numpy(), self.output[0].cpu().numpy()
                def canonical(rows):
                    return rows[np.lexsort(tuple(rows[:, i] for i in reversed(range(6))))]
                exact = np.array_equal(canonical(a), canonical(b))
                self.info['validation'].append(dict(input=record, finite=finite,
                    ordered_exact=ordered_exact, multiset_exact=bool(exact),cpu_stable_decoder_exact=bool(decoder_exact)))
                if not finite or not exact or not decoder_exact:
                    torch.save(dict(input=self.x.cpu(), eager=expected.cpu(), graph=self.output.cpu()), report_dir / 'gpu-mismatch.pt')
                    raise RuntimeError('448 graph/eager output mismatch; evidence saved')
            self.info['validation_passed'] = True
            write_json(report_dir / 'gpu-validation.json', self.info)
        except Exception as exc:
            write_json(report_dir / 'gpu-validation.json', dict(self.info, validation_passed=False, error=str(exc)))
            raise

    def prepare(self, x):
        self.host.copy_(self.torch.from_numpy(x))
        self.x.copy_(self.host, non_blocking=True)

    def infer(self):
        self.graph.replay()
        self.done.record()
        self.done.synchronize()

    def detections(self):
        return self.output[0].float().cpu().numpy().copy()

    def close(self):
        self.context.__exit__(None, None, None)
