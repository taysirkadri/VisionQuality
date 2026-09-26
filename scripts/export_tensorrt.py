"""Warm and verify an ORT/TensorRT engine cache for a fixed-size YOLO ONNX model."""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to a YOLOv8 ONNX model")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--cache", default="models/tensorrt")
    parser.add_argument("--workspace-mb", type=int, default=2048)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("Install the 'tensorrt' extra: pip install -e '.[tensorrt]'") from exc

    providers = ort.get_available_providers()
    if "TensorrtExecutionProvider" not in providers:
        raise SystemExit("TensorrtExecutionProvider is unavailable. Run scripts/verify_gpu.py.")

    import os
    os.makedirs(args.cache, exist_ok=True)
    options = {
        "device_id": args.device,
        "trt_fp16_enable": args.fp16,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": args.cache,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": args.cache,
        "trt_max_workspace_size": args.workspace_mb * 1024 * 1024,
    }
    session = ort.InferenceSession(
        args.model,
        providers=[
            ("TensorrtExecutionProvider", options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    input_name = session.get_inputs()[0].name
    dummy = np.zeros((args.batch, 3, args.height, args.width), dtype=np.float32)
    session.run(None, {input_name: dummy})
    print("providers=" + ",".join(session.get_providers()))
    print("engine_cache=" + args.cache)


if __name__ == "__main__":
    main()
