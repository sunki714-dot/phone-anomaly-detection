"""Configuration loading and device resolution."""

import yaml


def load_config(path="configs/default.yaml"):
    """Load a YAML config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device="auto"):
    """Resolve the compute device. ``"auto"`` picks CUDA when available."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
