# VisionQuality benchmark results

No hardware TensorRT result should be published until a real NVIDIA GPU, compatible
CUDA/TensorRT libraries, a labeled defect dataset, and a versioned ONNX/YOLO checkpoint
are available.

Run benchmarks with:

```powershell
python scripts/benchmark_inference.py models/best.onnx --provider cpu --requests 100
python scripts/benchmark_inference.py models/best.onnx --provider cuda --requests 100
python scripts/benchmark_inference.py models/best.onnx --provider tensorrt --fp16 --requests 100
```

Archive raw output with the following provenance:

- checkpoint SHA-256
- hardware and VRAM
- NVIDIA driver
- CUDA version
- TensorRT version
- ONNX Runtime version
- precision and batch size
- image dimensions
- warm/cold start status
- P50/P95/P99 latency
- throughput
- precision, recall, mAP, and false-negative rate
