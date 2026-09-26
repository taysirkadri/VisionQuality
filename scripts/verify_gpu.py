"""Verify NVIDIA/CUDA availability for VisionQuality TensorRT inference."""

from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> int:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        print("gpu=unavailable nvidia-smi=missing")
        return 1

    gpu_query = [
        nvidia_smi,
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    print(subprocess.check_output(gpu_query, text=True).strip())

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime=missing install onnxruntime-gpu")
        return 2

    providers = ort.get_available_providers()
    print("onnxruntime-providers=" + ",".join(providers))
    if "TensorrtExecutionProvider" not in providers:
        print("tensorrt=unavailable")
        return 3

    print("tensorrt=available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
