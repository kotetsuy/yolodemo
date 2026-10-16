# SPDX-License-Identifier: AGPL-3.0-only
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.model_utils import ROOT

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('backend', choices=['gpu', 'npu'])
    args = p.parse_args()
    folder = ROOT / 'results/model-validation'
    folder.mkdir(parents=True, exist_ok=True)
    if args.backend == 'gpu':
        from app.workers.gpu import GPU
        engine = GPU(folder)
    else:
        from app.workers.npu_output_sync import bootstrap_output_sync
        bootstrap_output_sync()
        from app.workers.npu import NPU
        engine = NPU(folder)
    print(engine.info)
    engine.close()
