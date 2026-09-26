"""Backend-neutral direct ONNX benchmark for warm TensorRT/CUDA/CPU sessions."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--provider", choices=["cpu", "cuda", "tensorrt"], default="cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cache", default="models/tensorrt")
    args = parser.parse_args()

    import os

    import onnxruntime as ort

    if args.provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        os.makedirs(args.cache, exist_ok=True)
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": args.fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": args.cache,
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": args.cache,
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    session = ort.InferenceSession(args.model, providers=providers)
    input_name = session.get_inputs()[0].name
    image = np.zeros((1, 3, args.size, args.size), dtype=np.float32)

    session.run(None, {input_name: image})
    latencies = []
    for _ in range(args.requests):
        started = time.perf_counter()
        session.run(None, {input_name: image})
        latencies.append((time.perf_counter() - started) * 1000)
    latencies.sort()

    print("providers=" + ",".join(session.get_providers()))
    print(f"requests={args.requests}")
    print(f"mean_ms={statistics.mean(latencies):.2f}")
    print(f"p50_ms={latencies[len(latencies) // 2]:.2f}")
    print(f"p95_ms={latencies[max(0, round(len(latencies) * 0.95) - 1)]:.2f}")


if __name__ == "__main__":
    main()
