"""Post-processing on top of the anomaly heatmap.

- :func:`phone_mask` — isolate the phone and drop the background (GrabCut).
- :func:`classify_defect` — bucket a heatmap into one of four defect types using
  only geometry (no labels needed).
- :func:`detect_stuck_pixel` — catch small coloured dead/stuck pixels that a
  smooth heatmap can miss.
"""

import cv2
import numpy as np
from PIL import Image

# defect key -> Korean display name
DEFECT_TYPES = {
    "point": "점결함(데드픽셀)",
    "horizontal": "가로 패널불량",
    "vertical": "세로 패널불량",
    "crack": "액정 균열",
}

# defect key -> condition grade used for pricing
DEFECT_GRADE = {
    "point": "B",
    "horizontal": "C",
    "vertical": "C",
    "crack": "C",
}


def _to_rgb_array(image, size):
    """Coerce a path / PIL image / array into a resized RGB ``np.uint8`` array."""
    if isinstance(image, (str, bytes)):
        pil = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = Image.fromarray(np.asarray(image)).convert("RGB")
    return np.array(pil.resize((size, size)))


def phone_mask(image, size=256):
    """Return a float mask (1 = phone, 0 = background).

    Uses GrabCut to segment the largest foreground blob; on failure falls back to
    a centred vertical box so the pipeline still runs.
    """
    img = _to_rgb_array(image, size)
    gc = np.zeros((size, size), np.uint8)
    rect = (int(size * 0.10), int(size * 0.10), int(size * 0.80), int(size * 0.80))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        m = np.where((gc == 1) | (gc == 3), 1, 0).astype(np.uint8)
        n, lbl, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:  # keep only the largest connected component
            m = (lbl == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        if 0.05 * size * size < m.sum() < 0.95 * size * size:
            return m.astype(np.float32)
    except Exception:
        pass
    m = np.zeros((size, size), np.float32)
    m[int(size * 0.12) : int(size * 0.88), int(size * 0.28) : int(size * 0.72)] = 1.0
    return m


def classify_defect(amap, rel_thr=0.45):
    """Classify a heatmap into a defect type using shape statistics only.

    Returns ``(defect_key, features_dict)``. Anisotropy separates horizontal vs
    vertical panel faults; small + compact reads as a point defect; otherwise a
    crack.
    """
    a = amap.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    mask = a > rel_thr
    H, W = mask.shape

    if mask.sum() < 4:
        return "point", {"area": 0.0, "spread": 0.0, "aniso": 0.0}

    area_ratio = float(mask.mean())
    ys, xs = np.where(mask)
    spread = max((ys.max() - ys.min() + 1) / H, (xs.max() - xs.min() + 1) / W)

    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    row_fill = float(mask[rows].mean(axis=1).mean())
    col_fill = float(mask[:, cols].mean(axis=0).mean())
    aniso = (row_fill - col_fill) / (row_fill + col_fill + 1e-9)

    feat = {
        "area": round(area_ratio, 3),
        "spread": round(spread, 2),
        "row_fill": round(row_fill, 3),
        "col_fill": round(col_fill, 3),
        "aniso": round(aniso, 2),
    }

    if aniso > 0.20:
        return "horizontal", feat
    if aniso < -0.20:
        return "vertical", feat
    if area_ratio < 0.03 and spread < 0.35:
        return "point", feat
    return "crack", feat


def detect_stuck_pixel(image, sat_thr=150, val_thr=110, min_a=2, max_a=60):
    """Detect a small, saturated dead/stuck pixel inside the phone region.

    Returns ``(found, (x, y) | None, area_px)`` in the original image's
    coordinates. Strict size/shape filters avoid firing on panel lines.
    """
    S = 256
    pil = image if isinstance(image, Image.Image) else Image.open(image)
    pil = pil.convert("RGB")
    img = np.array(pil.resize((S, S)))

    pm = phone_mask(pil, S).astype(np.uint8)
    pm = cv2.erode(pm, np.ones((7, 7), np.uint8))  # trim edges to avoid border hits

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    cand = ((hsv[:, :, 1] > sat_thr) & (hsv[:, :, 2] > val_thr) & (pm > 0)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    n, lbl, stats, cent = cv2.connectedComponentsWithStats(cand)
    spots = []
    for i in range(1, n):
        area = int(stats[i, 4])
        w_, h_ = int(stats[i, 2]), int(stats[i, 3])
        if not (min_a <= area <= max_a):
            continue
        if max(w_, h_) > 10:  # reject elongated streaks
            continue
        if area / (w_ * h_ + 1e-9) < 0.4:  # reject hollow shapes; keep dots
            continue
        cx = cent[i][0] * pil.width / S
        cy = cent[i][1] * pil.height / S
        spots.append((area, (int(cx), int(cy))))

    if spots:
        area, loc = max(spots, key=lambda s: s[0])
        return True, loc, area
    return False, None, 0
