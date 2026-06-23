"""Evaluate PatchCore (and optionally PaDiM + ensemble) and save diagnostic plots.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --out artifacts
"""

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # headless: write figures to files, no display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phone_anomaly.config import load_config, resolve_device
from phone_anomaly.data import prepare_dataset
from phone_anomaly.features import FeatureExtractor
from phone_anomaly.metrics import bootstrap_auroc, delong_test
from phone_anomaly.models import PaDiM, PatchCore, zfuse


def main():
    parser = argparse.ArgumentParser(description="Evaluate the anomaly detectors.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--no-padim", action="store_true", help="skip PaDiM + ensemble")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
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
    test_files = data["test_files"]
    test_labels = data["test_labels"]
    print(f"Train(normal) {len(data['train_good'])} | "
          f"Test {len(test_files)} ({int((test_labels == 0).sum())} normal / "
          f"{int((test_labels == 1).sum())} defect)")

    extractor = FeatureExtractor(
        backbone=cfg["backbone"]["name"],
        layers=cfg["backbone"]["layers"],
        img_size=cfg["img_size"],
        device=device,
    )

    # --- PatchCore ---------------------------------------------------------
    patchcore = PatchCore(
        extractor,
        coreset_ratio=cfg["patchcore"]["coreset_ratio"],
        n_projection=cfg["patchcore"]["n_projection"],
        smoothing_sigma=cfg["patchcore"]["smoothing_sigma"],
        seed=cfg["seed"],
    ).fit(data["train_good"])

    t0 = time.time()
    pc_scores = np.array([patchcore.score(p) for p in test_files])
    print(f"PatchCore inference: {len(test_files)} imgs | {time.time() - t0:.1f}s")

    auroc = roc_auc_score(test_labels, pc_scores)
    mean_auc, lo, hi = bootstrap_auroc(test_labels, pc_scores, seed=cfg["seed"])
    print(f"PatchCore AUROC = {auroc:.4f} (95% CI {lo:.3f}~{hi:.3f}, bootstrap)")

    # threshold from config (or F1-optimal as a fallback hint)
    prec, rec, thr = precision_recall_curve(test_labels, pc_scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_thr = cfg["patchcore"]["threshold"]
    pred = (pc_scores >= best_thr).astype(int)
    print(f"Threshold = {best_thr:.3f}")
    print("Confusion matrix [normal, defect]:")
    print(confusion_matrix(test_labels, pred))

    # diagnostic figure: ROC / F1-vs-threshold / score histogram
    fpr, tpr, _ = roc_curve(test_labels, pc_scores)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(fpr, tpr, lw=2, label=f"AUROC={auroc:.3f}")
    axs[0].plot([0, 1], [0, 1], "--", c="gray")
    axs[0].set_title("ROC Curve")
    axs[0].set_xlabel("FPR")
    axs[0].set_ylabel("TPR")
    axs[0].legend()
    axs[1].plot(thr, f1s[:-1], lw=2)
    axs[1].axvline(best_thr, c="r", ls="--", label=f"thr={best_thr:.2f}")
    axs[1].set_title("F1 vs threshold")
    axs[1].legend()
    axs[2].hist(pc_scores[test_labels == 0], bins=20, alpha=0.6, label="normal")
    axs[2].hist(pc_scores[test_labels == 1], bins=20, alpha=0.6, label="defect")
    axs[2].axvline(best_thr, c="r", ls="--")
    axs[2].set_title("Score distribution")
    axs[2].legend()
    fig.tight_layout()
    fig_path = os.path.join(args.out, "evaluation.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    # --- PaDiM + ensemble --------------------------------------------------
    if not args.no_padim:
        padim = PaDiM(extractor, n_dims=cfg["padim"]["n_dims"], seed=cfg["seed"])
        padim.fit(data["train_good"])
        pd_scores = np.array([padim.score(p) for p in test_files])
        auroc_pd = roc_auc_score(test_labels, pd_scores)

        ens = zfuse(pc_scores, pd_scores, cfg["ensemble"]["weight"])
        auroc_ens = roc_auc_score(test_labels, ens)
        print(f"PaDiM AUROC     = {auroc_pd:.4f}")
        print(f"Ensemble AUROC  = {auroc_ens:.4f}")

        (a_ens, a_pc), z, p = delong_test(test_labels, ens, pc_scores)
        sig = "significant" if p < 0.05 else "not significant"
        print(f"DeLong (Ensemble vs PatchCore): dAUROC={a_ens - a_pc:+.3f}, "
              f"z={z:.2f}, p={p:.4f} ({sig})")


if __name__ == "__main__":
    main()
