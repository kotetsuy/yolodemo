# MIGraphX GPU live demo validation (2026-10-31)

GPU worker: FP32 ONNX, fixed 1×3×448×448, MIGraphXExecutionProvider, DLPack IO binding, native HIP graph replay. Radeon 8060S; model SHA-256 `349a966a0b1948f8eb33e2679ee559b14d815dc336e9a9c2d031ffcc71c7d61a`.

30 seconds × 3 trials, playlist input and MJPEG client connected: 600.398, 610.527, 607.566 FPS. Pooled: **606.163 FPS**. GPU target: 600 FPS; passed: True.

Counts represent synchronized completed inferences on the latest available frame. Input video is about 30 FPS; repeated inference does not create additional input frames. The loop includes new input copies and sampled detection transfers while decoding, preprocessing and streaming run concurrently. This is a short measurement on this machine, not a sustained performance guarantee.

Eight startup validation inputs passed FP32 PyTorch detection comparison. All 21 existing tests passed. Raw local evidence: `results/benchmark-migraphx-live.json`; worker evidence: `results/demo-20261031-161806-60f1b3/gpu-validation.json`. Generated JSON files are excluded from Git.

Reproduce:

```bash
./start_all.sh --source playlist --no-browser
.venv/bin/python scripts/benchmark_demo.py --backends gpu --seconds 30 --repeats 3
```
