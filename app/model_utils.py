# SPDX-License-Identifier: AGPL-3.0-only
"""Model provenance, stable YOLO26 decode, and validation utilities.

Decode and smoke comparison follow ~/yolofast/benchmarks/npu_*.py.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
YOLOFAST = Path(os.environ.get('YOLOFAST_DIR', str(Path.home() / 'yolofast'))).expanduser().resolve()
WEIGHTS = YOLOFAST / 'models/yolo26n.pt'
MODEL = ROOT / 'models/npu/yolo26n-448-split-a16w8-perchannel-int16.onnx'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def decode(boxes, scores):
    boxes, scores = boxes[0].T, scores[0].T
    anchors = np.argsort(-scores.max(axis=1), kind='stable')[:300]
    flat = scores[anchors].reshape(-1)
    indices = np.argsort(-flat, kind='stable')[:300]
    return np.column_stack((boxes[anchors[indices // 80]], flat[indices], indices % 80)).astype(np.float32)


def score_output(scores, meta):
    transform = meta.get('score_transform')
    if not transform:
        return scores
    # Original final Sigmoid + INT16 Q/DQ, now outside the NPU graph.
    x = np.clip(scores, -80, 80)
    x = np.reciprocal(np.float32(1) + np.exp(-x))
    x = np.clip(np.rint(x / np.float32(transform['scale'])) + np.float32(transform['zero_point']), -32768, 32767)
    return (x-np.float32(transform['zero_point']))*np.float32(transform['scale'])


def qdq(x, params):
    scale,zero=np.float32(params['scale']),np.float32(params['zero_point'])
    return (np.clip(np.rint(x/scale)+zero,-32768,32767)-zero)*scale


def output_pair(raw, meta):
    classifiers=meta.get('cpu_classifiers')
    if not classifiers:
        return raw[0],score_output(raw[1],meta)
    logits=[]
    for feature,params in zip(raw[1:],classifiers):
        weights=np.asarray(params['weights'],np.float32)
        bias=np.asarray(params['bias'],np.float32)
        values=weights @ feature[0].reshape(weights.shape[1],-1)+bias[:,None]
        for stage in params['qdq_stages']:
            values=qdq(values,stage)
        logits.append(values)
    scores=qdq(np.concatenate(logits,axis=1)[None],meta['concat_qdq'])
    return raw[0],score_output(scores,meta)


def compare_detections(reference, candidate):
    rd, gd = reference[reference[:, 4] >= .25], candidate[candidate[:, 4] >= .25]
    valid = bool(np.isfinite(candidate).all() and np.all(gd[:, 2:4] >= gd[:, :2])
                 and np.all(gd[:, 5] == np.rint(gd[:, 5])))
    matches, used = [], set()
    for row in rd:
        choices = []
        for j, other in enumerate(gd):
            if j in used or row[5] != other[5]:
                continue
            wh = np.maximum(0, np.minimum(row[2:4], other[2:4]) - np.maximum(row[:2], other[:2]))
            inter = float(np.prod(wh))
            union = float(np.prod(row[2:4] - row[:2]) + np.prod(other[2:4] - other[:2]) - inter)
            choices.append((inter / max(union, 1e-9), j))
        if not choices:
            matches.append(dict(iou=0, score_diff=None))
            continue
        iou, j = max(choices)
        used.add(j)
        matches.append(dict(iou=iou, score_diff=float(abs(row[4] - gd[j, 4]))))
    passed = valid and len(rd) == len(gd) and all(m['iou'] >= .9 and m['score_diff'] is not None and m['score_diff'] <= .05 for m in matches)
    return dict(passed=bool(passed), geometry_valid=valid, reference_count=len(rd), candidate_count=len(gd), matches=matches)


def cases():
    import cv2
    from app.preprocess import preprocess, SIZE
    paths = [Path.home() / 'yolotest/calib_images/bus.jpg']
    paths += sorted((Path.home() / 'yolotest/calib2').glob('*.jpg'))[:2]
    if len(paths) != 3:
        raise RuntimeError('Three smoke images are required')
    result = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f'Cannot read smoke image: {path}')
        result.append((dict(path=str(path), sha256=sha256(path)), preprocess(image)[0]))
    for value in (0, 1):
        result.append((dict(synthetic=value), np.full((1, 3, SIZE, SIZE), value, np.float32)))
    for seed in (101, 202, 303):
        result.append((dict(seed=seed), np.random.default_rng(seed).random((1, 3, SIZE, SIZE), dtype=np.float32)))
    return result
