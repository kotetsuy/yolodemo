#!/usr/bin/env bash
# Source the installed Ryzen AI 1.9 environment with the verified library order.
source /opt/xilinx/xrt/setup.sh
source "$HOME/ryzenai_1_9/venv/bin/activate"
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:/lib/x86_64-linux-gnu:${RYZEN_AI_INSTALLATION_PATH}/onnxruntime/lib:${RYZEN_AI_INSTALLATION_PATH}/lib/python3.12/site-packages/lnx64.o/tools/peano/lib:${LD_LIBRARY_PATH:-}"
