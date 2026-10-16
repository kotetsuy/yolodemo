# YOLO26n GPU/NPU Demo

[English](README.md) | [日本語](READMEJ.md)

A local object detection demo for the Radeon GPU and Ryzen AI NPU. View live detections in Chrome, switch between backends with a two-position slider, and choose one of several connected USB cameras. Both backends use **448×448 input with batch size 1**, while the displayed video retains its original aspect ratio.

The demo requests **1920×1080 at 30 FPS** from the camera and shows inference FPS separately from input FPS. Models are generated locally and are not included in this repository.

## Related Projects

| Project | Relationship to this demo |
|---|---|
| [LLaVA-NPU](https://github.com/kotetsuy/LLaVA-NPU) | Reference for USB camera discovery, format negotiation, and camera selection. |
| [yolofast_pub](https://github.com/kotetsuy/yolofast_pub) | Public ONNX quantization and Ryzen AI NPU evaluation tools from the yolofast work on which this demo is based. |

The [yolofast_pub README](https://github.com/kotetsuy/yolofast_pub/blob/main/README.md) now covers both its standalone ONNX environment and **additional environments for yolodemo**. For this demo, follow its additional-environment section to prepare the GPU and ONNX export environments, then run `./create_models.sh --setup-quant` here. The standalone `~/yolofast/.venv` is a separate environment and is not used by this demo.

The documented demo paths match the scripts: GPU and export under `~/yolofast`, quantization under this repository's `.venv-quant`, and NPU setup at `~/ryzenai_1_8/setup_ryzenai_env.sh`. `YOLOFAST_DIR` changes the yolofast root only. Weights, the original 60-image calibration recipe, images under `~/yolotest`, and the NPU SDK still require separate preparation. The public ONNX image manifest is not a replacement for this demo's recipe. The newly documented independent CPU export environment must pass model generation and hardware validation before its outputs are adopted; it was not used for the previously recorded measurements.

LLaVA-NPU is a reference project and does not need to be running.

## Clone

```bash
git clone https://github.com/kotetsuy/yolodemo.git ~/yolodemo
cd ~/yolodemo
```

Prepare the external prerequisites below before creating models or starting the demo. If the destination already exists, review it before choosing a different checkout directory.

## Current Demo Prerequisites

This table matches the yolodemo additional-environment section in the upstream README. The standalone ONNX environment is separate. The verified hardware is an AMD Ryzen AI Max+ 395 with Radeon 8060S (gfx1151) and an XDNA2 NPU. The scripts currently expect the following existing environments and files. Each worker runs in its own process, keeping the GPU/NPU runtimes separate from the web virtual environment.

| Purpose | Environment |
|---|---|
| GPU | `~/yolofast/.venv-rocm10-py314`, PyTorch ROCm, Radeon 8060S |
| ONNX export | `~/yolofast/.venv-npu-export` |
| Quantization | `./.venv-quant`, AMD Quark |
| NPU | `~/ryzenai_1_8/setup_ryzenai_env.sh`, VitisAI EP, `/dev/accel/accel0` |
| Weights | `~/yolofast/models/yolo26n.pt` |
| Calibration recipe | `~/yolofast/distribute/models/npu/accuracy95/perchannel/yolo26n-640-a16w8.json` and its specified 60 images |
| Web | `uv`, Python 3.12, `requirements.txt`, Chrome |
| Native build tools | A C++ compiler toolchain (for example, Ubuntu `build-essential`); Ninja is installed in `.venv-quant` |

Set `YOLOFAST_DIR` to change the yolofast location. On another machine, prepare compatible runtimes, weights, calibration images, and the required directory layout first. The model creation script checks prerequisites; it does not provision the GPU/NPU environments. Do not replace the VitisAI SDK environment with the CPU `onnxruntime` package from another environment. The demo creates 448 models under `models/npu/` without overwriting the existing 640 models.

## Creating Models Locally

Run the entire model preparation workflow through one script:

```bash
./stop_all.sh
./create_models.sh --setup-quant
./create_models.sh --check
./create_models.sh
./start_all.sh
```

`--check` verifies the Python environments, weights, calibration and smoke-test images, and NPU device/runtime without installing packages or generating models. Run without options to create the local quantization and web environments as needed, install their requirements, export YOLO26n at 448×448, split the box/score outputs, quantize with A16W8 per-channel MinMax/CLE, convert the input to INT16, and create the NPU backbone plus CPU classification head. The script validates the NPU, selects it only after passing the strict criteria, and then validates the GPU. It uses the Python helpers under `scripts/` internally; no manual stage-by-stage commands are required.

Prerequisites are the external environments in the table above, `~/yolofast/models/yolo26n.pt`, the original quantization recipe and its 60 calibration images with matching hashes, and `~/yolotest/calib_images/bus.jpg` plus at least two JPG files in `~/yolotest/calib2/`. The script does **not** download weights, datasets, or install GPU/NPU runtimes. Obtain them under their respective licenses before running it. `YOLOFAST_DIR` can override the yolofast location; yolotest and the Ryzen AI setup retain the locations listed above. Recipe image paths must resolve on this machine.

All outputs under `models/`, including ONNX files, CPU heads, provenance JSON and `selection.json`, remain local and are excluded from Git. Validation reports are under `results/model-validation/`. Existing base models are reused only if provenance hashes match; the selected candidate is revalidated on hardware. Modified or incomplete files cause failure; move the affected generated files aside before regenerating them. A failed NPU validation is not ignored or retried automatically. The score-difference limit remains 0.05.

`./create_models.sh --help` shows usage. The previous `scripts/prepare_models.sh` entry point forwards to this script. Startup also invokes the script if `selection.json` is missing. To reproduce the model files from a fresh checkout, supply the prerequisites above and run `./create_models.sh` before starting the demo.

### Local quantization environment

`./create_models.sh --setup-quant` creates `.venv-quant` in this repository using Python 3.12.13 installed under `.python/`. It installs the versions recorded in `requirements-quant.txt`, including AMD Quark 0.11.2 and CPU-only PyTorch 2.6.0. Downloads require network access on the first setup. The normal model creation command also performs this setup automatically. Keep `.python/` alongside the venv; do not copy an existing venv from another project.

Quantization no longer uses `~/yolotest/.venv-quant`. Calibration and smoke-test images still come from `~/yolotest`, and export/GPU/NPU environments retain their documented locations. Both `.venv-quant/` and `.python/` are excluded from Git; `requirements-quant.txt` is included so the environment can be recreated. The old environment remains available to its original project. Quark may compile its CPU custom operators on first import, so a native C++ compiler toolchain must also be available.

## Starting and Stopping

```bash
cd ~/yolodemo
./start_all.sh
```

After preparing the prerequisites and models above, Chrome opens `http://127.0.0.1:8000`, and the server runs in the background. On the first run, the script creates the web virtual environment and generates missing 448 models using `create_models.sh`. Both workers validate and warm up their models at startup. Initial NPU compilation can take some time.

Startup completes NPU validation first, then initializes the GPU. Camera/video capture and inference start after both are ready. The backend slider is disabled during preparation and becomes available when both workers are ready. Alternating-input output mismatches also occurred with the NPU alone; the output synchronization fix below addresses that failure.

```bash
./stop_all.sh
```

Stopping shuts down this project's server and inference workers and releases its shared memory. Starting an already running demo does not create a second instance; stopping an already stopped demo exits without further action. The stop script also handles demos started through `scripts/run_demo.sh`. Chrome remains open. Logs are appended to `.cache/run/demo.log`.

To run in the foreground and stop with Ctrl+C, use `bash scripts/run_demo.sh`.

```bash
# Start with playlist video and the NPU, without using a camera
./start_all.sh --source playlist --backend npu
# Start without opening the browser automatically
./start_all.sh --no-browser
# Prefer a specific camera; fall back to video if it cannot open or disconnects
./start_all.sh --camera /dev/video0
```

By default, cameras take priority. If no usable camera is available, the demo falls back to the configured local playlist. Prepare a playlist before relying on this fallback; videos are not included in Git.

## Camera Input

The “映像入力” (Video input) selector lets you choose **automatic selection, any connected camera, or the video playlist**. Even with multiple cameras connected, only one camera's video is displayed at a time. A manually selected camera takes priority. Selecting “自動” (Automatic) restores the configured priority order.

Following the camera handling in [LLaVA-NPU](https://github.com/kotetsuy/LLaVA-NPU), [camera.yaml](camera.yaml) defines camera priorities using stable USB by-id names and VID:PID identifiers. By default, the demo **requests 1920×1080 at 30 FPS and tries MJPG followed by YUYV**. If unsupported, it uses the resolution and format accepted by the camera and displays the actual dimensions and configured FPS. A configured 30 FPS and the measured input FPS are separate values. Inference remains at 448×448.

The camera list refreshes every second and includes only devices capable of video capture. In automatic mode, the demo switches to a preferred camera after its connection has remained stable for one second. The current video continues until the replacement camera delivers its first frame. If the selected camera disconnects, the demo moves to another available camera, or to video if none is available, and detects reconnections.

Edit `preferred`, `fallback: any/none`, `preempt`, `preempt_settle_sec`, and `format` in `camera.yaml` to customize this behavior. `fallback: none` restricts automatic selection to cameras in the preferred list. Cameras outside that list can still be selected manually.

```bash
# Override resolution and configured FPS at startup (stop first if running)
./start_all.sh --camera-width 2560 --camera-height 1440 --camera-fps 30
# Use a different camera configuration
./start_all.sh --camera-config /path/to/camera.yaml
# Reduce streaming load
./start_all.sh --display-width 1280
```

## Local Video Input

The development setup used a playlist of 10 short 2560×1440 videos. The entire `videos/` directory is excluded from Git, so a fresh checkout does not include those clips, `playlist.m3u8`, or `SOURCES.md`.

To use your own videos, create a local M3U playlist containing one file path per line. Relative paths are resolved from the playlist's directory; lines starting with `#` are treated as comments.

```bash
mkdir -p videos
# Place videos/example.mp4 here before starting
printf '%s\n' '#EXTM3U' 'example.mp4' > videos/playlist.m3u8
./start_all.sh --source playlist --playlist videos/playlist.m3u8
```

The default playlist is `videos/playlist.m3u8`. Use `--playlist /path/to/playlist.m3u8` to select another one. The default streaming width is 1920 pixels, configurable with `--display-width`. Keep video sources and licensing information in your local `videos/SOURCES.md`; the UI's “動画の出典” (Video sources) link reads that file. If you retain the original development clips, retain their source and license notices as well.

## What FPS Means

The prominent FPS value is **the number of completed inferences divided by measured elapsed seconds**. Because inference repeats on the latest frame, inference FPS can exceed 300 even when input is 30 FPS. It does not count new video frames. The value updates approximately every 0.3 seconds.

GPU timing includes synchronization. NPU timing includes INT16 quantization, NPU execution, the CPU classification head, and Top-k. Video decoding, letterboxing, and streaming run concurrently in separate threads, and their load is present during live measurements. Input FPS is shown separately. Inference FPS is left blank during input loss, preparation, or switching.

Backend switching waits for the previous worker to stop processing, updates the epoch, and discards old results. Workers are not restarted during switching. Frames and detections are matched by ID, and retained frames and streamed images are bounded.

## Inference and Numerical Validation

The GPU uses an FP16 HIP graph and increments its counter after GPU synchronization completes. Top-k results with equal scores are ordered stably by index without changing score values. At startup, eight inputs are used to check agreement between ordinary forward execution, graph replay, and the CPU's stable Top-k decoder.

The NPU retains yolofast's **A16W8, per-channel, MinMax, and CLE** settings and its 60 calibration images. It preserves the QDQ parameters recalibrated at 448, runs the backbone and box inference on the NPU, and runs the classification head and its preceding SiLU on the CPU. This arrangement improved the score differences that exceeded the acceptance criteria in the full-NPU configuration. The displayed loop speed includes the CPU head, INT16 input quantization, and Top-k.

The reference is CPU execution of the original 448 quantized model. The criteria remain **a threshold of 0.25, matching detection counts and classes, IoU≥0.9, and score differences≤0.05**. The selected model passed on three real images and five synthetic inputs, with a maximum score difference of 0.046207. Startup also checks reproducibility with alternating inputs. This validation is not an mAP measurement.

SHA-256 hashes are checked for the model, CPU head, and reference model. The selected model is recorded in `models/npu/selection.json`; NPU compilation caches are stored by model hash under `.cache/npu-448/`.

The NPU worker also synchronizes output buffers with `XCL_BO_SYNC_BO_FROM_DEVICE` immediately after NPU completion and before CPU depadding. A local compatibility library implements this fix for the tested Ryzen AI 1.8 / XRT 2.25.37 build. The NPU entry point checks the runtime library hashes, builds the library under `.cache/native/npu-output-sync/` using `g++` and `/opt/xilinx/xrt/include`, and re-executes the NPU process once with that library preloaded. SDK and driver files are unchanged. Every inference verifies that the synchronization ran; each runner invocation releases its retained buffer handles before returning. Startup reports record the library hashes, synchronization counts, and outstanding buffer count under `output_sync`.

The verified runtime fingerprints are in `app/workers/npu_output_sync.py`. An SDK/XRT update requires compatibility validation before updating those fingerprints; an unrecognized build stops with an explicit error. Custom NPU-only entry points must call `bootstrap_output_sync()` before opening devices or IPC; the demo worker and `scripts/validate_models.py npu` already do this. GPU and web processes do not inherit the NPU preload.

To repeat the hardware regression across five NPU session lifetimes and 10,000 alternating-input inferences:

```bash
source ~/ryzenai_1_8/setup_ryzenai_env.sh
PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 python scripts/check_npu_output_sync.py
```

This checks exact raw-output and detection equality, the original startup validation, synchronization coverage, and buffer release after each session. Results are saved to `results/npu-permanent-fix/regression/report.json`.

## Validation and Performance Measurement

```bash
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest -q
# Run after starting the demo in another terminal
.venv/bin/python scripts/benchmark_demo.py --output results/benchmark-formal.json
.venv/bin/python scripts/benchmark_demo.py --seconds 300 --repeats 1 --output results/benchmark-stress.json
# Automated Chrome validation (validation dependency)
uv pip install --python .venv/bin/python playwright
.venv/bin/python scripts/check_browser.py
```

Performance measurements use the selected backend's actual counters while also receiving MJPEG video. Formal benchmarks and continuous operation results are summarized in `results/VALIDATION.md`. Each run saves its environment, numerical validation, switching history, and shutdown state under `results/demo-<timestamp>-<ID>/`.

See [results/cameras/README.md](results/cameras/README.md) for high-resolution camera and input selection validation. Capture at 1920×1080, Chrome display, and input switching were verified with one physical USB camera. Camera priorities, disconnections, and reconnections were tested with simulated devices. Two physical cameras and physical disconnection have not been tested.

## Source Layout

- `app/cameras.py`, `camera.yaml`: USB camera inventory, priorities, and V4L2 format configuration.
- `app/capture.py`, `app/preprocess.py`: Input capture, real-time playback, 448 preprocessing, and coordinate mapping back to the original video.
- `app/ipc.py`, `app/coordinator.py`: IPC across Python environments, epochs, worker pause confirmation, rendering, and streaming.
- `app/workers/`: GPU/NPU validation, synchronized inference, and actual completion counters.
- `app/server.py`, `app/static/`: Localhost HTTP server and Chrome UI.
- `create_models.sh`, `scripts/prepare_models.py`, `scripts/refine_npu.py`: 448 model generation using the same quantization method and CPU head separation.
- `scripts/benchmark_demo.py`, `scripts/check_browser.py`: Performance measurement and validation in real Chrome.

## HTTP API

| Endpoint | Purpose / request body |
|---|---|
| `GET /api/status` | Backend, worker, input, and FPS status. |
| `GET /api/cameras` | Camera inventory and current input status. |
| `POST /api/backend` | `{"backend":"gpu"}` or `{"backend":"npu"}`. |
| `POST /api/camera` | `{"camera":"auto"}`, `{"camera":"playlist"}`, or `{"camera":"<camera-id>"}` using an ID from the camera inventory. |
| `GET /frame.jpg` | Latest detection image. |
| `GET /stream.mjpg` | Continuous MJPEG video. |
| `GET /source.zip` | Application source archive. |
| `GET /license` | AGPL-3.0 license text. |

## Files Excluded from Git

Generated models and their metadata, runtime caches, raw validation results, logs, and screenshots are kept locally and excluded by `.gitignore`. They may contain local paths or camera images. Written reports under `results/` are included; links to raw artifacts refer to files from a local run. Model artifacts can be regenerated with `create_models.sh`.

## License

This application, including its original code, scripts, configuration and documentation, is distributed under the **GNU Affero General Public License, version 3 only (AGPL-3.0-only)**. See [LICENSE](LICENSE) and [NOTICE](NOTICE). No warranty is provided.

Apache-2.0 notices for material adapted from LLaVA-NPU are preserved in [NOTICE](NOTICE) and [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt). Third-party libraries, model weights, datasets and SDKs retain their respective terms. Ultralytics YOLO26 code and model derivatives remain subject to its [AGPL-3.0 / Enterprise terms](https://www.ultralytics.com/license); models are not included in Git.

The UI provides links to the license and a ZIP archive of the application source served by the running instance. If you modify or deploy this application, keep that source offer available and include any additional corresponding source required for your modified version. The source archive contains application code, scripts, tests, configuration, documentation and license notices; it excludes local models, images, runtime logs, credentials and environments. It is not a bundled model or SDK distribution.
