# SPDX-License-Identifier: AGPL-3.0-only
"""Generate and validate 448 assets using the existing yolofast recipe."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache/ultralytics'))
from app.model_utils import WEIGHTS, YOLOFAST, MODEL, sha256, write_json, cases, decode
from app.preprocess import SIZE, preprocess, quantize
import numpy as np

BASE = MODEL.parent / 'yolo26n-448.onnx'
SPLIT = MODEL.parent / 'yolo26n-448-split-fp32.onnx'
QUANT = MODEL.parent / 'yolo26n-448-split-a16w8-perchannel.onnx'
RECIPE = YOLOFAST / 'distribute/models/npu/accuracy95/perchannel/yolo26n-640-a16w8.json'


def complete(path):
    if not path.exists():
        return False
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta.get('status') != 'complete' or meta.get('model_sha256') != sha256(path):
        raise RuntimeError(f'Incomplete or modified model: {path}. Move it aside before regenerating.')
    if meta.get('weights_sha256') != sha256(WEIGHTS) or meta.get('size') != SIZE:
        raise RuntimeError(f'Model provenance changed: {path}')
    if meta.get('recipe_sha256') != sha256(RECIPE):
        raise RuntimeError(f'Quantization recipe changed: {path}')
    return True


def metadata(path, **extra):
    write_json(path.with_suffix('.json'), dict(status='complete', size=SIZE,
        model_sha256=sha256(path), weights_sha256=sha256(WEIGHTS), recipe_sha256=sha256(RECIPE), **extra))


def export():
    import torch
    import onnx
    from ultralytics import YOLO
    if complete(BASE):
        return
    torch.set_num_threads(4)
    model = YOLO(str(WEIGHTS)).model.eval().cpu().float()
    model.end2end = True
    model.fuse(verbose=False)
    head = model.model[-1]
    head.export, head.format, head.dynamic = True, 'onnx', False
    class Forward(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)
    torch.manual_seed(42)
    x = torch.rand(1, 3, SIZE, SIZE)
    with torch.inference_mode():
        y = model(x)
        assert tuple(y.shape) == (1, 300, 6) and bool(y.isfinite().all())
        temp = BASE.with_name(BASE.stem + '.tmp.onnx')
        torch.onnx.export(Forward(), x, str(temp), input_names=['images'], output_names=['detections'],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(temp))
    temp.replace(BASE)
    metadata(BASE, torch=torch.__version__, opset=17, names=model.names)


def cpu(model):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    return ort.InferenceSession(model if isinstance(model, bytes) else str(model), so, providers=['CPUExecutionProvider'])


def split():
    import onnx
    if complete(SPLIT):
        return
    m = onnx.load(BASE)
    names = ['/model/model.23/Mul_2_output_0', '/model/model.23/Sigmoid_output_0']
    anchors = sum((SIZE // s) ** 2 for s in (8, 16, 32))
    shapes = [[1, 4, anchors], [1, 80, anchors]]
    producers = {v: n for n in m.graph.node for v in n.output}
    for name, op, shape in zip(names, ['Mul', 'Sigmoid'], shapes):
        if name not in producers or producers[name].op_type != op:
            raise RuntimeError(f'Unexpected head node: {name}')
        for v in [*m.graph.input, *m.graph.output, *m.graph.value_info]:
            if v.name == name:
                dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
                if dims and all(dims) and dims != shape:
                    raise RuntimeError(f'Unexpected shape: {dims}')
        if not any(v.name == name for v in m.graph.value_info):
            m.graph.value_info.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
    diagnostic = copy.deepcopy(m)
    diagnostic.graph.output.extend(onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, s) for n, s in zip(names, shapes))
    extracted = onnx.utils.Extractor(m).extract_model(['images'], names)
    onnx.checker.check_model(extracted)
    old, new = cpu(diagnostic.SerializeToString()), cpu(extracted.SerializeToString())
    checks = []
    from app.model_utils import compare_detections
    for record, x in cases():
        a, b = old.run(None, {'images': x}), new.run(None, {'images': x})
        exact = all(np.array_equal(u, v) and np.isfinite(v).all() for u, v in zip(a[-2:], b))
        check = dict(input=record, raw_exact=bool(exact), detections=compare_detections(a[0][0], decode(*b)))
        checks.append(check)
        if not exact or not check['detections']['passed']:
            write_json(SPLIT.with_suffix('.failed.json'), dict(checks=checks))
            raise RuntimeError('FP32 split verification failed')
    onnx.save(extracted, SPLIT)
    metadata(SPLIT, source_sha256=sha256(BASE), checks=checks, output_shapes=shapes)


def quant():
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config.config import Config
    from quark.onnx.quantization.config.custom_config import A16W8_CONFIG
    import cv2
    import importlib.metadata
    if complete(QUANT):
        return
    original = json.loads(RECIPE.read_text())
    records = original['calibration']
    if len(records) != 60:
        raise RuntimeError('Expected the original 60 calibration images')
    for r in records:
        r['path'] = os.path.expandvars(r['path'])
        if sha256(r['path']) != r['sha256']:
            raise RuntimeError(f'Calibration image changed: {r["path"]}')
    class Reader:
        def __init__(self):
            self.paths = iter(r['path'] for r in records)
        def get_next(self):
            p = next(self.paths, None)
            if p is None:
                return None
            im = cv2.imread(p)
            if im is None:
                raise RuntimeError(f'Cannot read {p}')
            return {'images': preprocess(im)[0]}
    config = copy.deepcopy(A16W8_CONFIG)
    config.per_channel = True
    assert config.include_cle
    from onnxruntime.quantization.calibrate import CalibrationMethod
    assert config.calibrate_method == CalibrationMethod.MinMax
    temp = QUANT.with_name(QUANT.stem + '.tmp.onnx')
    ModelQuantizer(Config(global_quant_config=config)).quantize_model(str(SPLIT), str(temp), Reader())
    import onnx
    onnx.checker.check_model(onnx.load(temp))
    temp.replace(QUANT)
    metadata(QUANT, source_sha256=sha256(SPLIT), calibration=records, recipe='A16W8_CONFIG',
             per_channel=True, cle=True, calibration_method='MinMax', quark=importlib.metadata.version('amd-quark'))


def int16():
    import onnx
    from onnx import numpy_helper
    if complete(MODEL):
        return
    m = onnx.load(QUANT)
    consumers = [n for n in m.graph.node if 'images' in n.input]
    if len(consumers) != 1 or consumers[0].op_type != 'QuantizeLinear':
        raise RuntimeError('Expected one input QuantizeLinear')
    q = consumers[0]
    init = {x.name: numpy_helper.to_array(x) for x in m.graph.initializer}
    scale, zero = init[q.input[1]], init[q.input[2]]
    assert scale.size == zero.size == 1 and zero.dtype == np.int16 and float(scale.item()) > 0
    m.graph.node.remove(q)
    for n in m.graph.node:
        for i, name in enumerate(n.input):
            if name == q.output[0]:
                n.input[i] = 'images'
    m.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.INT16
    for v in list(m.graph.value_info):
        if v.name == q.output[0]:
            m.graph.value_info.remove(v)
    onnx.checker.check_model(m)
    old, new = cpu(QUANT), cpu(m.SerializeToString())
    scratch = np.empty((1, 3, SIZE, SIZE), np.float32)
    buf = np.empty(scratch.shape, np.int16)
    checks = []
    for record, x in cases():
        a = old.run(None, {'images': x})
        b = new.run(None, {'images': quantize(x, scale.item(), zero.item(), scratch, buf)})
        exact = all(np.array_equal(u, v) and np.isfinite(v).all() for u, v in zip(a, b))
        checks.append(dict(input=record, raw_exact=bool(exact)))
        if not exact:
            raise RuntimeError('INT16 conversion is not equivalent')
    onnx.save(m, MODEL)
    metadata(MODEL, scale=float(scale.item()), zero_point=int(zero.item()), input_type='INT16',
             reference_model=str(QUANT), reference_sha256=sha256(QUANT), checks=checks,
             per_channel=True, recipe='A16W8_CONFIG', output_shapes=[[1,4,4116],[1,80,4116]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['export', 'split', 'quant', 'int16'])
    args = parser.parse_args()
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    globals()[args.stage]()
    print(f'{args.stage}: complete', flush=True)
