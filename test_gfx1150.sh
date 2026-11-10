#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Run the GPU preflight and the existing numerical validation on gfx1150.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  printf '%s\n' 'Usage: ./test_gfx1150.sh' \
    'Set YOLOFAST_DIR to the yolofast directory if it is not ~/yolofast.' \
    'Requires the MIGraphX environment, model, weights and smoke-test images.' \
    'Runs GPU inference and writes results/model-validation/gpu-validation.json.'
  exit 0
fi
if [[ $# != 0 ]]; then
  echo 'Usage: ./test_gfx1150.sh' >&2
  exit 2
fi

export YOLOFAST_DIR="${YOLOFAST_DIR:-$HOME/yolofast}"
export PYTHONPATH="$PWD"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
export OPENBLAS_NUM_THREADS=1
task_python="$YOLOFAST_DIR/.venv-migraphx/bin/python"
if [[ ! -x "$task_python" ]]; then
  echo "Missing GPU Python environment: $task_python" >&2
  exit 1
fi

echo 'Checking ROCm GPU and MIGraphX runtime...'
"$task_python" - <<'PY'
import torch
import onnxruntime as ort

if not torch.version.hip or not torch.cuda.is_available():
    raise SystemExit('ROCm GPU is unavailable to PyTorch')
props = torch.cuda.get_device_properties(0)
arch = props.gcnArchName.split(':', 1)[0]
print(f'GPU: {props.name} ({props.gcnArchName})', flush=True)
print(f'PyTorch: {torch.__version__}; HIP: {torch.version.hip}', flush=True)
print(f'ONNX Runtime providers before backend initialization: {ort.get_available_providers()}', flush=True)
if arch != 'gfx1150':
    raise SystemExit(f'Expected gfx1150, detected {arch}')
# Ultralytics may register the MIGraphX plugin during backend initialization.
# The GPU worker verifies the actual session provider after that initialization.
PY

echo 'Running GPU model and numerical validation...'
"$task_python" scripts/validate_models.py gpu
echo 'GPU validation passed: results/model-validation/gpu-validation.json'
