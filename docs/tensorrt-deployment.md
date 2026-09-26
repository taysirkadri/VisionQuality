# TensorRT deployment

## CPU-only demo

```powershell
pip install -e ".[dev]"
$env:DEFECT_BACKEND="onnx"
uvicorn industrial_defect_detection.api:app --host 0.0.0.0 --port 8000
```

## NVIDIA TensorRT

1. Install an NVIDIA driver compatible with the selected CUDA runtime.
2. Install Docker with the NVIDIA Container Toolkit.
3. Place an approved ONNX model at `models/best.onnx`.
4. Verify host GPU support:

```powershell
python scripts/verify_gpu.py
```

5. Warm the engine cache:

```powershell
python scripts/export_tensorrt.py models/best.onnx --fp16 --cache models/tensorrt
```

6. Start GPU Compose:

```powershell
docker compose -f docker/compose.gpu.yaml up --build
```

The service reports active ONNX Runtime providers through `GET /providers`
and `GET /healthz`. TensorRT mode does not silently switch a missing real model
to the fake detector: configuration failures are reported as degraded health.

## Benchmark

```powershell
python scripts/benchmark_inference.py models/best.onnx --provider tensorrt --fp16 --requests 100
python scripts/benchmark_inference.py models/best.onnx --provider cpu --requests 100
```

Record driver, CUDA, TensorRT, ONNX Runtime, hardware, image size, precision,
throughput, P50/P95/P99 latency, and accuracy metrics. Engine files and caches
must not be committed.
