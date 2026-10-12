# SPDX-License-Identifier: AGPL-3.0-only
"""Select only the strictly validated, fixed-recipe NPU candidate."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.model_utils import ROOT,sha256,write_json

if __name__=='__main__':
    validation=ROOT/'results/model-validation/npu-validation.json'
    data=json.loads(validation.read_text())
    model=ROOT/'models/npu/candidates/yolo26n-448-cpu-head-silu-outside-int16.onnx'
    if not data.get('validation_passed') or data.get('model_sha256')!=sha256(model):
        raise SystemExit('NPU candidate has not passed hardware validation')
    if not all(v['passed'] for v in data['validation']):
        raise SystemExit('NPU smoke criterion failed')
    if not all(m['score_diff']<=.05 and m['iou']>=.9 for v in data['validation'] for m in v['matches']):
        raise SystemExit('NPU acceptance criteria changed')
    write_json(ROOT/'models/npu/selection.json',dict(model=str(model.relative_to(ROOT)),sha256=sha256(model),
        validation=str(validation.relative_to(ROOT)),score_tolerance=.05))
    print(f'Selected NPU model: {model}')
