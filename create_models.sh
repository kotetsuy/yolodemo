#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Single entry point for local model export, quantization and hardware validation.
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  --help|-h)
    cat <<'HELP'
Usage: ./create_models.sh [--check | --setup-quant]

Creates the local .venv-quant quantization environment.
Uses existing export/GPU environments in ~/yolofast and NPU setup in ~/ryzenai_1_8.
YOLOFAST_DIR may override ~/yolofast. Does not download weights or datasets.
--check checks prerequisites without installing packages or creating models.
--setup-quant installs only the local quantization environment and then exits.
Stop the demo before building. Generated artifacts remain local under models/.
HELP
    exit 0 ;;
  --check) task_check_only=1; task_setup_only=0 ;;
  --setup-quant) task_check_only=0; task_setup_only=1 ;;
  '') task_check_only=0; task_setup_only=0 ;;
  *) echo 'Usage: ./create_models.sh [--check | --setup-quant]' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Too many arguments' >&2; exit 2; }
export PYTHONPATH="$PWD"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
export OPENBLAS_NUM_THREADS=1
export YOLOFAST_DIR="${YOLOFAST_DIR:-$HOME/yolofast}"
task_export_python="$YOLOFAST_DIR/.venv-npu-export/bin/python"
task_quant_python="$PWD/.venv-quant/bin/python"
command -v uv >/dev/null || { echo 'Missing uv' >&2; exit 1; }
if [[ "$task_check_only" == 0 ]]; then
  export UV_PYTHON_INSTALL_DIR="$PWD/.python"
  export UV_CACHE_DIR="$PWD/.cache/uv"
  if [[ ! -x "$task_quant_python" ]]; then
    uv python install 3.12.13 --no-bin
    uv venv --managed-python --python 3.12.13 .venv-quant
  fi
  # Use the CPU-only PyTorch wheel required by the original Quark environment.
  uv pip install --python "$task_quant_python" --no-deps \
    --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
  uv pip install --python "$task_quant_python" -r requirements-quant.txt
fi
if [[ ! -x "$task_quant_python" ]]; then
  echo 'Missing local .venv-quant. Run ./create_models.sh --setup-quant first.' >&2
  exit 1
fi
export PATH="$PWD/.venv-quant/bin:$PATH"
if [[ "$task_setup_only" == 1 ]]; then
  "$task_quant_python" -c 'from quark.onnx import ModelQuantizer; import cv2, onnx, onnxruntime; print("Local quantization environment: OK")'
  exit 0
fi
task_gpu_python="$YOLOFAST_DIR/.venv-rocm10-py314/bin/python"
task_npu_setup="$HOME/ryzenai_1_8/setup_ryzenai_env.sh"
for task_executable in "$task_export_python" "$task_quant_python" "$task_gpu_python"; do
  [[ -x "$task_executable" ]] || { echo "Missing Python environment: $task_executable" >&2; exit 1; }
done
[[ -f "$task_npu_setup" ]] || { echo "Missing NPU setup: $task_npu_setup" >&2; exit 1; }
command -v uv >/dev/null || { echo 'Missing uv' >&2; exit 1; }
"$task_export_python" -c 'import torch, ultralytics, onnx'
"$task_gpu_python" -c 'import torch, ultralytics'
"$task_quant_python" - <<'PY'
import json, os
from pathlib import Path
import cv2, onnx, onnxruntime, quark
from app.model_utils import YOLOFAST, WEIGHTS, cases, sha256
recipe=YOLOFAST/'distribute/models/npu/accuracy95/perchannel/yolo26n-640-a16w8.json'
if not WEIGHTS.is_file():
    raise SystemExit(f'Missing YOLO26n weights: {WEIGHTS}')
records=json.loads(recipe.read_text())['calibration']
if len(records)!=60:
    raise SystemExit('The recipe must specify 60 calibration images')
for record in records:
    path=Path(os.path.expandvars(record['path']))
    if not path.is_file() or sha256(path)!=record['sha256']:
        raise SystemExit(f'Missing or modified calibration image: {path}')
cases()  # Three real smoke images and five synthetic inputs.
print('Weights, 60 calibration images, smoke images and Python dependencies: OK')
PY
(
  source "$task_npu_setup" >/dev/null
  python - <<'PY'
from pathlib import Path
import onnxruntime as ort
if not list(Path('/dev/accel').glob('accel*')):
    raise SystemExit('NPU device unavailable')
if 'VitisAIExecutionProvider' not in ort.get_available_providers():
    raise SystemExit('VitisAIExecutionProvider unavailable')
print('NPU device and runtime: OK')
PY
)
if [[ "$task_check_only" == 1 ]]; then exit 0; fi
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
uv pip install --python .venv/bin/python -r requirements.txt
for task_stage in export split quant int16; do
  echo "Model stage: $task_stage"
  if [[ "$task_stage" == export ]]; then
    "$task_export_python" scripts/prepare_models.py "$task_stage"
  else
    "$task_quant_python" scripts/prepare_models.py "$task_stage"
  fi
done
task_candidate="$PWD/models/npu/candidates/yolo26n-448-cpu-head-silu-outside-int16.onnx"
if [[ ! -f "$task_candidate" ]]; then
  "$task_quant_python" scripts/refine_npu.py cpu-head-silu-outside
fi
# Validate the NPU alone before selecting this candidate. Strict criteria remain in force.
(
  source "$task_npu_setup" >/dev/null
  export NPU_MODEL="$task_candidate"
  python scripts/validate_models.py npu
)
.venv/bin/python scripts/select_npu_model.py
"$task_gpu_python" scripts/validate_models.py gpu
printf '%s\n' 'Models ready. Generated models, metadata and validation reports are excluded from Git.'
