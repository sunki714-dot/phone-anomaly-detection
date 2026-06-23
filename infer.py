"""Score a single image from the command line.

Usage:
    python infer.py path/to/photo.jpg
    python infer.py path/to/photo.jpg --memory-bank artifacts/memory_bank.pt
"""

import argparse
import os

from phone_anomaly.config import load_config, resolve_device
from phone_anomaly.data import prepare_dataset
from phone_anomaly.features import FeatureExtractor
from phone_anomaly.models import PatchCore
from phone_anomaly.postprocess import DEFECT_TYPES, classify_defect, phone_mask


def load_detector(cfg, device, memory_bank_path=None):
    extractor = FeatureExtractor(
        backbone=cfg["backbone"]["name"],
        layers=cfg["backbone"]["layers"],
        img_size=cfg["img_size"],
        device=device,
    )
    if memory_bank_path and os.path.exists(memory_bank_path):
        return PatchCore.load(memory_bank_path, extractor)
    data = prepare_dataset(
        cfg["data"]["good_dir"],
        cfg["data"]["defect_dir"],
        train_ratio=cfg["data"]["train_ratio"],
        use_demo_if_missing=cfg["data"]["use_demo_if_missing"],
        seed=cfg["seed"],
    )
    return PatchCore(
        extractor,
        coreset_ratio=cfg["patchcore"]["coreset_ratio"],
        n_projection=cfg["patchcore"]["n_projection"],
        smoothing_sigma=cfg["patchcore"]["smoothing_sigma"],
        seed=cfg["seed"],
    ).fit(data["train_good"])


def main():
    parser = argparse.ArgumentParser(description="Score one image for defects.")
    parser.add_argument("image", help="path to the image to score")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--memory-bank", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    detector = load_detector(cfg, device, args.memory_bank)

    amap, score = detector.anomaly_map(args.image, mask_fn=phone_mask)
    threshold = cfg["patchcore"]["threshold"]
    is_defect = score >= threshold

    print(f"image    : {args.image}")
    print(f"score    : {score:.3f} (threshold {threshold:.3f})")
    print(f"verdict  : {'DEFECT' if is_defect else 'NORMAL'}")
    if is_defect:
        key, feat = classify_defect(amap)
        print(f"type     : {DEFECT_TYPES[key]}")
        print(f"features : {feat}")


if __name__ == "__main__":
    main()
