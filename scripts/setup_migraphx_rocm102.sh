#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv" PYTHONNOUSERSITE=1
unset PYTHONPATH
python312="${PYTHON_FOR_MIGRAPHX:-$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12}"
venv="$PWD/.venv-migraphx-rocm102"
if [[ ! -x "$venv/bin/python" ]]; then uv venv --python "$python312" "$venv"; fi
uv pip install --python "$venv/bin/python" --index https://stable.repo.amd.com/rocm/whl-next/ \
  'torch[device-gfx1151]==2.13.0+rocm10.2.0' 'torchvision[device-gfx1151]==0.28.0+rocm10.2.0'
# Pin torch so Ultralytics cannot replace the ROCm build.
uv pip install --python "$venv/bin/python" --index https://stable.repo.amd.com/rocm/whl-next/ \
  --constraint scripts/migraphx-rocm102-constraints.txt 'ultralytics==8.4.194' 'onnx==1.23.2' 'onnxslim==0.1.98' 'flatbuffers==25.12.19'
uv pip install --python "$venv/bin/python" --no-deps \
  --index https://stable.repo.amd.com/rocm/onnxruntime/whl-next/ \
  --index https://stable.repo.amd.com/rocm/migraphx/whl-next/ \
  'onnxruntime-ep-migraphx==1.0.0+rocm10.2.0' 'migraphx-libs==2.19.0+rocm10.2.0' 'onnxruntime==1.29.0'
uv pip check --python "$venv/bin/python"
"$venv/bin/python" - <<'PY'
import importlib.metadata as m
import torch, onnxruntime
from ultralytics.nn.backends.onnx import ONNXBackend
assert torch.__version__ == '2.13.0+rocm10.2.0', torch.__version__
for name in ('onnxruntime-ep-migraphx', 'migraphx-libs'):
    print(name, m.version(name))
print('torch', torch.__version__, 'onnxruntime', onnxruntime.__version__)
PY
