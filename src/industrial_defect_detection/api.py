"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from industrial_defect_detection.config import Settings
from industrial_defect_detection.inference import Detector, InferenceError, create_detector
from industrial_defect_detection.schemas import DetectionResponse
from industrial_defect_detection.service import DetectionService
from industrial_defect_detection.vision import ImageDecodeError


def create_app(detector_factory: Callable[[Settings], Detector] = create_detector) -> FastAPI:
    """Create an API whose model loads once, during application startup."""
    settings = Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.load_model()
        yield

    app = FastAPI(
        title="Industrial Defect Detection API",
        version="0.1.0",
        description="YOLOv8 defect detection over PyTorch or ONNX Runtime.",
        lifespan=lifespan,
    )
    app.state.service = None
    app.state.model_error = None
    app.state.metrics = {
        "requests_total": 0,
        "requests_success": 0,
        "requests_error": 0,
        "detections_total": 0,
        "latency_seconds_total": 0.0,
    }
    app.state.metrics_lock = Lock()

    def load_model() -> None:
        try:
            app.state.service = DetectionService(detector_factory(settings))
            app.state.model_error = None
        except (InferenceError, ValueError) as exc:
            # Keep /healthz available so orchestrators receive a useful error.
            app.state.service = None
            app.state.model_error = str(exc)

    app.state.load_model = load_model

    @app.get("/healthz", tags=["operations"])
    def healthz() -> dict[str, str | bool]:
        detector = app.state.service.detector if app.state.service else None
        return {
            "status": "ok" if app.state.service else "degraded",
            "model_loaded": app.state.service is not None,
            "model_backend": getattr(detector, "backend", "none"),
            "detail": app.state.model_error or "Model is ready.",
        }

    @app.get("/metrics", tags=["operations"])
    def metrics() -> dict[str, int | float]:
        with app.state.metrics_lock:
            values = dict(app.state.metrics)
        count = values["requests_total"]
        values["latency_seconds_average"] = (
            values["latency_seconds_total"] / count if count else 0.0
        )
        return values

    @app.get("/", tags=["operations"])
    def root() -> dict[str, str]:
        return {"service": "industrial-defect-detection", "docs": "/docs"}

    @app.post("/v1/detect", response_model=DetectionResponse, tags=["inference"])
    async def detect(
        image: UploadFile = File(..., description="JPEG, PNG, or another OpenCV-decodable image."),
        annotate: bool = Query(False, description="Include an annotated JPEG as base64."),
        confidence: float | None = Query(
            None, ge=0.01, le=0.99, description="Reserved for request-scoped thresholding."
        ),
    ) -> DetectionResponse:
        del (
            confidence
        )  # Threshold is intentionally configured at process start for reproducibility.
        if app.state.service is None:
            raise HTTPException(
                status_code=503, detail=app.state.model_error or "Model is unavailable."
            )
        payload = await image.read(settings.max_upload_bytes + 1)
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {settings.max_upload_bytes} byte upload limit.",
            )
        started = perf_counter()
        try:
            result = app.state.service.inspect_bytes(payload, include_annotation=annotate)
        except ImageDecodeError as exc:
            elapsed = perf_counter() - started
            with app.state.metrics_lock:
                app.state.metrics["requests_total"] += 1
                app.state.metrics["requests_error"] += 1
                app.state.metrics["latency_seconds_total"] += elapsed
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except InferenceError as exc:
            elapsed = perf_counter() - started
            with app.state.metrics_lock:
                app.state.metrics["requests_total"] += 1
                app.state.metrics["requests_error"] += 1
                app.state.metrics["latency_seconds_total"] += elapsed
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        elapsed = perf_counter() - started
        with app.state.metrics_lock:
            app.state.metrics["requests_total"] += 1
            app.state.metrics["requests_success"] += 1
            app.state.metrics["detections_total"] += len(result.detections)
            app.state.metrics["latency_seconds_total"] += elapsed
        return result

    return app


app = create_app()
