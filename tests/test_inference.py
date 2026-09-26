from __future__ import annotations

import numpy as np

from industrial_defect_detection.config import Settings
from industrial_defect_detection.inference import (
    InferenceError,
    TensorRTDetector,
    _decode_yolov8_output,
    _preprocess,
)


def test_preprocess_letterboxes_to_nchw_rgb_float() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor, scale, pad_x, pad_y = _preprocess(image, 640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert scale == 3.2
    assert pad_x == 0
    assert pad_y == 160


def test_decode_yolov8_output_filters_and_unletterboxes() -> None:
    # (1, 4 + 2 classes, 2 candidates); the second candidate is below confidence.
    output = np.array([[[320, 320], [320, 320], [128, 100], [64, 100], [0.9, 0.1], [0.1, 0.2]]])
    detections = _decode_yolov8_output(
        output=output,
        scale=2.0,
        pad_x=0,
        pad_y=100,
        image_shape=(270, 320, 3),
        confidence_threshold=0.5,
        iou_threshold=0.45,
        labels={0: "crack", 1: "scratch"},
    )

    assert len(detections) == 1
    assert detections[0].label == "crack"
    assert detections[0].confidence == 0.9
    assert detections[0].box.x1 == 128
    assert detections[0].box.y1 == 94


def test_tensorrt_configuration_defaults(tmp_path) -> None:
    settings = Settings(
        backend="onnx-tensorrt",
        model_path=tmp_path / "missing.onnx",
        confidence=0.35,
        iou=0.45,
        image_size=640,
        max_upload_bytes=10 * 1024 * 1024,
        class_names=("scratch",),
    )

    settings.validate()
    assert settings.tensorrt_fp16 is True
    assert settings.tensorrt_device_id == 0
    assert settings.max_batch_size == 1


def test_tensorrt_missing_model_does_not_fallback_to_fake(tmp_path) -> None:
    settings = Settings(
        backend="onnx-tensorrt",
        model_path=tmp_path / "missing.onnx",
        confidence=0.35,
        iou=0.45,
        image_size=640,
        max_upload_bytes=10 * 1024 * 1024,
        class_names=("scratch",),
    )

    import pytest

    with pytest.raises(InferenceError, match="was not found"):
        TensorRTDetector.load(settings)
