"""Small HTTP benchmark with no external models or datasets."""

from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the defect API on synthetic images.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", dest="request_count", type=int, default=100)
    parser.add_argument("--size", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.request_count < 1:
        raise SystemExit("--requests must be positive")

    health = requests.get(f"{args.url.rstrip('/')}/healthz", timeout=10)
    health.raise_for_status()
    image = np.full((args.size, args.size, 3), 128, dtype=np.uint8)
    cv2.circle(image, (args.size // 2, args.size // 2), args.size // 8, (32, 32, 32), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise SystemExit("Could not create benchmark image")

    latencies: list[float] = []
    detections = 0
    endpoint = f"{args.url.rstrip('/')}/v1/detect"
    for _ in range(args.request_count):
        started = time.perf_counter()
        response = requests.post(
            endpoint,
            files={"image": ("benchmark.jpg", encoded.tobytes(), "image/jpeg")},
            timeout=30,
        )
        response.raise_for_status()
        latencies.append(time.perf_counter() - started)
        detections += len(response.json()["detections"])

    latencies.sort()
    print(f"backend={health.json().get('model_backend')}")
    print(f"requests={args.request_count} detections={detections}")
    print(f"mean_ms={statistics.mean(latencies) * 1000:.2f}")
    print(f"p50_ms={latencies[len(latencies) // 2] * 1000:.2f}")
    print(f"p95_ms={latencies[max(0, round(len(latencies) * 0.95) - 1)] * 1000:.2f}")


if __name__ == "__main__":
    main()
