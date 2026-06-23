"""Dataset utilities: listing images, generating demo data, and splitting.

Convention: train only on **normal** images (정상). The test set mixes the
held-out normals with all **defect** images (결함), labelled 0 (normal) / 1
(defect).
"""

import os
import random

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(directory):
    """Return a sorted list of image paths in ``directory`` (empty if missing)."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def make_demo(out_dir="data/demo", n_good=80, n_defect=30, size=256, seed=42):
    """Generate synthetic good/defect images so the pipeline runs without real data.

    Good images are flat dark panels; defects add a few bright scratch lines.
    Returns ``(good_dir, defect_dir)``.
    """
    import cv2

    good_dir = os.path.join(out_dir, "good")
    defect_dir = os.path.join(out_dir, "defect")
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(defect_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    def clean():
        img = np.zeros((size, size, 3), np.uint8)
        c = int(rng.integers(30, 70))
        img[:] = (c, c, c + 10)
        img[20:-20, 40:-40] = (c + 15, c + 15, c + 25)
        return img

    for i in range(n_good):
        Image.fromarray(clean()).save(os.path.join(good_dir, f"g{i:03d}.png"))

    for i in range(n_defect):
        img = clean()
        for _ in range(int(rng.integers(1, 4))):
            x1, y1 = rng.integers(40, size - 40, 2)
            x2 = int(x1 + rng.integers(-40, 40))
            y2 = int(y1 + rng.integers(-40, 40))
            cv2.line(img, (int(x1), int(y1)), (x2, y2), (220, 220, 230), int(rng.integers(1, 3)))
        Image.fromarray(img).save(os.path.join(defect_dir, f"d{i:03d}.png"))

    return good_dir, defect_dir


def prepare_dataset(
    good_dir,
    defect_dir,
    train_ratio=0.8,
    use_demo_if_missing=True,
    demo_dir="data/demo",
    seed=42,
):
    """Build train/test splits.

    If the real directories are missing and ``use_demo_if_missing`` is True,
    synthetic demo data is generated instead.

    Returns a dict with ``train_good``, ``test_files``, ``test_labels`` and the
    resolved ``good_dir`` / ``defect_dir``.
    """
    if not (os.path.isdir(good_dir) and os.path.isdir(defect_dir)):
        if use_demo_if_missing:
            good_dir, defect_dir = make_demo(demo_dir, seed=seed)
        else:
            raise FileNotFoundError(
                f"Missing data directories: {good_dir!r} / {defect_dir!r}"
            )

    good = list_images(good_dir)
    defect = list_images(defect_dir)
    if not good:
        raise RuntimeError(f"No images found in {good_dir!r}")

    rng = random.Random(seed)
    rng.shuffle(good)
    n_train = int(len(good) * train_ratio)

    train_good = good[:n_train]
    test_good = good[n_train:]
    test_files = test_good + defect
    test_labels = np.array([0] * len(test_good) + [1] * len(defect))

    return {
        "train_good": train_good,
        "test_files": test_files,
        "test_labels": test_labels,
        "good_dir": good_dir,
        "defect_dir": defect_dir,
    }
