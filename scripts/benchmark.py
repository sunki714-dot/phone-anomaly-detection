"""Ablations: backbone comparison, coreset-ratio trade-off, and seed variance.

Usage:
    python scripts/benchmark.py --config configs/default.yaml
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phone_anomaly.config import load_config, resolve_device
from phone_anomaly.data import prepare_dataset
from phone_anomaly.features import FeatureExtractor, available_backbones
from phone_anomaly.models import PatchCore


def _score_all(detector, files):
    return np.array([detector.score(p) for p in files])


def backbone_comparison(data, cfg, device):
    print("\n=== Backbone comparison (PatchCore, coreset 10%) ===")
    rows = []
    for name in available_backbones():
        try:
            extractor = FeatureExtractor(backbone=name, img_size=cfg["img_size"], device=device)
            detector = PatchCore(extractor, coreset_ratio=0.10, seed=cfg["seed"]).fit(
                data["train_good"]
            )
            t0 = time.time()
            scores = _score_all(detector, data["test_files"])
            auroc = roc_auc_score(data["test_labels"], scores)
            rows.append((name, round(auroc, 3), round(time.time() - t0, 1)))
        except Exception as exc:  # noqa: BLE001 - report and continue
            rows.append((name, "err", str(exc)[:30]))
    print(f"{'backbone':<18}{'AUROC':<10}{'infer(s)':<10}")
    for name, auroc, t in rows:
        print(f"{name:<18}{str(auroc):<10}{str(t):<10}")


def coreset_ablation(data, cfg, device):
    print("\n=== Coreset-ratio ablation ===")
    extractor = FeatureExtractor(
        backbone=cfg["backbone"]["name"], img_size=cfg["img_size"], device=device
    )
    print(f"{'ratio':<10}{'bank':<10}{'AUROC':<10}{'infer(s)':<10}")
    for ratio in (0.05, 0.10, 0.25):
        detector = PatchCore(extractor, coreset_ratio=ratio, seed=cfg["seed"]).fit(
            data["train_good"]
        )
        t0 = time.time()
        scores = _score_all(detector, data["test_files"])
        auroc = roc_auc_score(data["test_labels"], scores)
        bank = detector.memory_bank.shape[0]
        print(f"{f'{int(ratio*100)}%':<10}{bank:<10}{round(auroc, 3):<10}{round(time.time() - t0, 1):<10}")


def seed_variance(data, cfg, device, n=5):
    print(f"\n=== Coreset seed variance (n={n}) ===")
    extractor = FeatureExtractor(
        backbone=cfg["backbone"]["name"], img_size=cfg["img_size"], device=device
    )
    aurocs = []
    for sd in range(n):
        detector = PatchCore(extractor, coreset_ratio=0.10, seed=sd).fit(data["train_good"])
        scores = _score_all(detector, data["test_files"])
        aurocs.append(roc_auc_score(data["test_labels"], scores))
    print(f"AUROC = {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Run anomaly-detection ablations.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    print(f"Device: {device}")

    data = prepare_dataset(
        cfg["data"]["good_dir"],
        cfg["data"]["defect_dir"],
        train_ratio=cfg["data"]["train_ratio"],
        use_demo_if_missing=cfg["data"]["use_demo_if_missing"],
        seed=cfg["seed"],
    )

    backbone_comparison(data, cfg, device)
    coreset_ablation(data, cfg, device)
    seed_variance(data, cfg, device)


if __name__ == "__main__":
    main()
