# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-23

### Added
- Initial release, refactored from the original research notebook into an
  installable package.
- `FeatureExtractor`: frozen ImageNet backbone (ResNet18 / WideResNet50-2 /
  DenseNet121) producing per-patch descriptors.
- `PatchCore`: coreset memory bank with nearest-neighbour anomaly scoring and
  spatial heatmaps; supports `save()` / `load()`.
- `PaDiM`: per-position Gaussian baseline with Mahalanobis scoring.
- `zfuse`: z-score ensemble of PatchCore and PaDiM scores.
- Post-processing: phone-region masking (GrabCut), label-free geometric defect
  classification, and stuck-pixel detection.
- Resale-price estimation from model, age, and condition grade.
- Metrics: bootstrap AUROC confidence interval and DeLong significance test.
- CLI scripts: `train.py`, `evaluate.py`, `benchmark.py`, `infer.py`.
- Gradio demo (`app.py`) for end-to-end diagnosis and pricing.
- Synthetic demo-data generator so the pipeline runs without real images.
