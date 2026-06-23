"""Build the PatchCore memory bank from normal images and save it.

Usage:
    python scripts/train.py --config configs/default.yaml --out artifacts/memory_bank.pt
"""

import argparse
import os
import sys
import time

# allow running as a standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phone_anomaly.config import load_config, resolve_device
from phone_anomaly.data import prepare_dataset
from phone_anomaly.features import FeatureExtractor
from phone_anomaly.models import PatchCore


def main():
    parser = argparse.ArgumentParser(description="Fit and save a PatchCore memory bank.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="artifacts/memory_bank.pt")
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
    print(f"Training on {len(data['train_good'])} normal images.")

    extractor = FeatureExtractor(
        backbone=cfg["backbone"]["name"],
        layers=cfg["backbone"]["layers"],
        img_size=cfg["img_size"],
        device=device,
    )
    detector = PatchCore(
        extractor,
        coreset_ratio=cfg["patchcore"]["coreset_ratio"],
        n_projection=cfg["patchcore"]["n_projection"],
        smoothing_sigma=cfg["patchcore"]["smoothing_sigma"],
        seed=cfg["seed"],
    )

    t0 = time.time()
    detector.fit(data["train_good"])
    detector.save(args.out)
    print(
        f"Memory bank: {tuple(detector.memory_bank.shape)} "
        f"| saved to {args.out} | {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
