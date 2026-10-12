#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
if ! .venv/bin/python -c 'import numpy, cv2, pyudev, yaml' >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r requirements.txt
fi
if [[ ! -f models/npu/selection.json ]]; then
  ./create_models.sh
fi
export PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -m app.server "$@"
