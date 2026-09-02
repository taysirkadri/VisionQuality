from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from industrial_defect_detection.api import create_app
from industrial_defect_detection.config import Settings
from industrial_defect_detection.inference import create_detector
from industrial_defect_detection.schemas import BoundingBox, Detection


class FakeDetector:
    backend = "fake"

    def detect(self, image: np.ndarray) -> list[Detection]:
        del image
        return [
            Detection(
                class_id=1,
                label="inclusion",
                confidence=0.88,
                box=BoundingBox(x1=2, y1=3, x2=20, y2=30),
            )
        ]


def _app_with_fake_detector():
    app = create_app(detector_factory=lambda _settings: FakeDetector())
    app.state.load_model()
    return app


def _endpoint(app, path: str):
    return next(route.endpoint for route in app.routes if route.path == path)


class FakeUpload:
    """A minimal async upload used when calling FastAPI handlers directly."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    async def read(self, size: int = -1) -> bytes:
        return self.payload if size < 0 else self.payload[:size]


def _upload(payload: bytes) -> FakeUpload:
    return FakeUpload(payload)


def _resolve_immediately(coroutine):
    """Drive a handler whose only await is the in-memory fake upload."""
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("The fake upload unexpectedly suspended the handler.")


def test_health_and_detection_with_annotation() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    app = _app_with_fake_detector()
    health = _endpoint(app, "/healthz")()
    response = _resolve_immediately(
        _endpoint(app, "/v1/detect")(_upload(encoded.tobytes()), annotate=True, confidence=None)
    )

    assert health["status"] == "ok"
    assert response.model_backend == "fake"
    assert response.detections[0].label == "inclusion"
    assert response.annotated_image_base64


def test_detection_rejects_malformed_image() -> None:
    app = _app_with_fake_detector()
    with pytest.raises(HTTPException) as error:
        _resolve_immediately(
            _endpoint(app, "/v1/detect")(_upload(b"broken"), annotate=False, confidence=None)
        )

    assert error.value.status_code == 422
    assert "supported image" in error.value.detail


def test_metrics_track_success_and_detections() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    app = _app_with_fake_detector()
    _resolve_immediately(
        _endpoint(app, "/v1/detect")(_upload(encoded.tobytes()), annotate=False, confidence=None)
    )

    metrics = _endpoint(app, "/metrics")()
    assert metrics["requests_total"] == 1
    assert metrics["requests_success"] == 1
    assert metrics["detections_total"] == 1


def test_real_backend_missing_model_falls_back_to_fake() -> None:
    settings = Settings(
        backend="ultralytics",
        model_path=Settings.from_env().model_path,
        confidence=0.35,
        iou=0.45,
        image_size=640,
        max_upload_bytes=10 * 1024 * 1024,
        class_names=("scratch",),
    )
    detector = create_detector(settings)
    assert detector.backend == "fake"
