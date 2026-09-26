# Model card

## Intended use

This service is a starting point for visual inspection of manufactured surfaces.
It detects localized visual anomalies in still images and returns labeled boxes
and confidence scores. It is intended to support—not replace—trained quality
inspectors and established safety controls.

## Model and classes

The repository intentionally ships **without weights** and defaults to a
dependency-free fake demo detector. Supply a versioned YOLOv8 `.pt` model or an
ONNX export through `DEFECT_MODEL_PATH`; missing weights/dependencies trigger
the fake fallback. The default labels are `crazing`, `inclusion`, `patches`, `pitted_surface`,
`rolled-in_scale`, and `scratches`; replace them when using another dataset.

## Evaluation requirements

Evaluate a candidate model on a held-out sample from every line, camera,
material grade, lighting regime, and defect severity where it will operate.
Report per-class precision, recall, mAP, false-negative rate, latency, and
calibration. Set acceptance thresholds together with process owners.

For TensorRT/CUDA deployment, compare the same weights in FP32 PyTorch, CPU ONNX,
CUDA ONNX, and TensorRT FP16 modes. Record hardware, NVIDIA driver, CUDA, TensorRT,
ONNX Runtime versions, P50/P95/P99 latency, throughput, VRAM, accuracy delta, mAP,
precision, recall, false-negative rate, and cold-engine-build time. Do not infer that
a faster backend is safer; retained model accuracy and false-negative controls govern
release decisions.

## Limitations and operational safeguards

- Confidence is not a probability of product fitness or safety.
- Domain shift (camera position, glare, contamination, texture, or new lots)
  can sharply reduce accuracy.
- False negatives can be costly. Route low-confidence or no-detection cases
  according to a documented human-review policy.
- Record model version, threshold, image source, and reviewer decision for
  traceability.
- Revalidate after retraining, lens/lighting changes, or software upgrades.
- TensorRT engine output can change with GPU, driver, ONNX graph, precision, and engine
  cache contents. Rebenchmark and revalidate every version change.
- TensorRT FP16 and INT8 quantization require precision-specific accuracy testing.
- Generated engine and timing caches are hardware-specific and must not be committed.

