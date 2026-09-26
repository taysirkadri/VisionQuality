"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CLASS_NAMES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings for an inference process."""

    backend: str
    model_path: Path
    confidence: float
    iou: float
    image_size: int
    max_upload_bytes: int
    class_names: tuple[str, ...]
    tensorrt_fp16: bool = True
    tensorrt_device_id: int = 0
    tensorrt_workspace_mb: int = 2048
    tensorrt_engine_cache: Path = Path("models/tensorrt")
    tensorrt_timing_cache: Path = Path("models/tensorrt")
    provider_fallback: bool = True
    warmup_requests: int = 10
    max_batch_size: int = 1

    @classmethod
    def from_env(cls) -> Settings:
        raw_names = os.getenv("DEFECT_CLASS_NAMES", "")
        names = tuple(name.strip() for name in raw_names.split(",") if name.strip())
        return cls(
            backend=os.getenv("DEFECT_BACKEND", "fake").lower(),
            model_path=Path(os.getenv("DEFECT_MODEL_PATH", "models/best.pt")),
            confidence=float(os.getenv("DEFECT_CONFIDENCE", "0.35")),
            iou=float(os.getenv("DEFECT_IOU", "0.45")),
            image_size=int(os.getenv("DEFECT_IMAGE_SIZE", "640")),
            max_upload_bytes=int(os.getenv("DEFECT_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
            class_names=names or DEFAULT_CLASS_NAMES,
            tensorrt_fp16=os.getenv("DEFECT_TENSORRT_FP16", "true").lower() in {"1", "true", "yes"},
            tensorrt_device_id=int(os.getenv("DEFECT_TENSORRT_DEVICE_ID", "0")),
            tensorrt_workspace_mb=int(os.getenv("DEFECT_TENSORRT_MAX_WORKSPACE_MB", "2048")),
            tensorrt_engine_cache=Path(
                os.getenv("DEFECT_TENSORRT_ENGINE_CACHE", "models/tensorrt")
            ),
            tensorrt_timing_cache=Path(
                os.getenv("DEFECT_TENSORRT_TIMING_CACHE", "models/tensorrt")
            ),
            provider_fallback=os.getenv("DEFECT_PROVIDER_FALLBACK", "true").lower()
            in {"1", "true", "yes"},
            warmup_requests=int(os.getenv("DEFECT_WARMUP_REQUESTS", "10")),
            max_batch_size=int(os.getenv("DEFECT_MAX_BATCH_SIZE", "1")),
        )

    def validate(self) -> None:
        if self.backend not in {"fake", "ultralytics", "onnx", "onnx-tensorrt", "tensorrt"}:
            raise ValueError(
                "DEFECT_BACKEND must be 'fake', 'ultralytics', 'onnx', "
                "'onnx-tensorrt', or 'tensorrt'"
            )
        if not 0 < self.confidence < 1:
            raise ValueError("DEFECT_CONFIDENCE must be between 0 and 1")
        if not 0 < self.iou < 1:
            raise ValueError("DEFECT_IOU must be between 0 and 1")
        if self.image_size < 32:
            raise ValueError("DEFECT_IMAGE_SIZE must be at least 32")
        if self.max_upload_bytes < 1:
            raise ValueError("DEFECT_MAX_UPLOAD_BYTES must be positive")
        if self.tensorrt_device_id < 0:
            raise ValueError("DEFECT_TENSORRT_DEVICE_ID must be zero or greater")
        if self.tensorrt_workspace_mb < 64:
            raise ValueError("DEFECT_TENSORRT_MAX_WORKSPACE_MB must be at least 64")
        if self.warmup_requests < 0:
            raise ValueError("DEFECT_WARMUP_REQUESTS must be zero or greater")
        if self.max_batch_size < 1:
            raise ValueError("DEFECT_MAX_BATCH_SIZE must be at least 1")
