"""Anomaly detectors built on frozen features.

- :class:`PatchCore` — coreset memory bank of normal patches; anomaly = distance
  to the nearest memory vector. Produces a spatial heatmap.
- :class:`PaDiM` — per-position multivariate Gaussian of normal patches; anomaly
  = Mahalanobis distance.
- :func:`zfuse` — z-score fusion to ensemble two score vectors.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


class PatchCore:
    """PatchCore-style detector: nearest-neighbour distance to a coreset bank.

    Train on normal images only via :meth:`fit`, then score new images with
    :meth:`score` (scalar) or :meth:`anomaly_map` (heatmap + scalar).
    """

    def __init__(self, extractor, coreset_ratio=0.10, n_projection=128, smoothing_sigma=2.0, seed=42):
        self.extractor = extractor
        self.coreset_ratio = coreset_ratio
        self.n_projection = n_projection
        self.sigma = smoothing_sigma
        self.seed = seed
        self.memory_bank = None
        self.fmap = None

    @staticmethod
    def coreset_subsample(feats, ratio=0.10, n_proj=128, seed=42):
        """Greedy farthest-point coreset over a random projection of ``feats``.

        Keeps roughly ``ratio`` of the patches while preserving coverage, which
        shrinks the memory bank (and inference cost) with little accuracy loss.
        """
        n = feats.shape[0]
        m = max(1, int(n * ratio))
        rng = np.random.default_rng(seed)
        proj = torch.randn(feats.shape[1], n_proj) / np.sqrt(n_proj)
        z = feats @ proj
        sel = [int(rng.integers(n))]
        min_d = torch.cdist(z, z[sel]).squeeze(1)
        for _ in range(m - 1):
            idx = int(torch.argmax(min_d))
            sel.append(idx)
            min_d = torch.minimum(min_d, torch.cdist(z, z[idx : idx + 1]).squeeze(1))
        return feats[sel]

    def fit(self, train_items):
        """Build the memory bank from normal images."""
        feats, self.fmap = self.extractor.patch_features(train_items)
        bank = self.coreset_subsample(feats, self.coreset_ratio, self.n_projection, self.seed)
        self.memory_bank = bank.to(self.extractor.device)
        return self

    def _check_fitted(self):
        if self.memory_bank is None:
            raise RuntimeError("PatchCore is not fitted. Call .fit(...) or .load(...) first.")

    @torch.no_grad()
    def _patch_distances(self, items):
        """Nearest-memory distance for every patch -> (flat_distances, fmap)."""
        self._check_fitted()
        feats, fmap = self.extractor.patch_features(items)
        feats = feats.to(self.extractor.device)
        dmin = []
        for i in range(0, feats.shape[0], 4096):
            d = torch.cdist(feats[i : i + 4096], self.memory_bank)
            dmin.append(d.min(1).values)
        return torch.cat(dmin).cpu().numpy(), fmap

    def score(self, item):
        """Image-level anomaly score (max smoothed patch distance)."""
        dmin, (H, W) = self._patch_distances([item])
        amap = gaussian_filter(dmin.reshape(H, W), sigma=self.sigma)
        return float(amap.max())

    def anomaly_map(self, item, mask_fn=None):
        """Return ``(heatmap, score)`` upsampled to ``img_size``.

        ``mask_fn(item, size) -> 2D array`` optionally zeroes out the background
        (e.g. everything outside the phone) so edges don't trigger false alarms.
        """
        dmin, (H, W) = self._patch_distances([item])
        amap = gaussian_filter(dmin.reshape(H, W), sigma=self.sigma)
        size = self.extractor.img_size
        up = F.interpolate(
            torch.tensor(amap)[None, None].float(),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        if mask_fn is not None:
            up = up * mask_fn(item, size)
        return up, float(up.max())

    def save(self, path):
        """Persist the memory bank and hyper-parameters to ``path``."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._check_fitted()
        torch.save(
            {
                "memory_bank": self.memory_bank.cpu(),
                "fmap": self.fmap,
                "coreset_ratio": self.coreset_ratio,
                "n_projection": self.n_projection,
                "smoothing_sigma": self.sigma,
                "seed": self.seed,
                "backbone": self.extractor.backbone_name,
            },
            path,
        )

    @classmethod
    def load(cls, path, extractor):
        """Restore a detector from :meth:`save`, attaching a fresh ``extractor``."""
        ckpt = torch.load(path, map_location="cpu")
        obj = cls(
            extractor,
            coreset_ratio=ckpt["coreset_ratio"],
            n_projection=ckpt["n_projection"],
            smoothing_sigma=ckpt["smoothing_sigma"],
            seed=ckpt["seed"],
        )
        obj.memory_bank = ckpt["memory_bank"].to(extractor.device)
        obj.fmap = ckpt["fmap"]
        return obj


class PaDiM:
    """PaDiM detector: per-position Gaussian, scored by Mahalanobis distance."""

    def __init__(self, extractor, n_dims=100, seed=42):
        self.extractor = extractor
        self.n_dims = n_dims
        self.seed = seed
        self.mean = None
        self.inv_cov = None
        self.sel = None

    def fit(self, train_items):
        """Estimate the mean and inverse covariance per spatial position."""
        grid, _ = self.extractor.grid_features(train_items)  # (N, P, C)
        N, P, C = grid.shape
        d = min(self.n_dims, C)
        gen = torch.Generator().manual_seed(self.seed)
        self.sel = torch.randperm(C, generator=gen)[:d]
        tr = grid[:, :, self.sel].numpy()
        self.mean = tr.mean(0)
        reg = np.eye(d) * 0.01
        cov = np.empty((P, d, d), np.float32)
        for p in range(P):
            cov[p] = np.cov(tr[:, p, :], rowvar=False) + reg
        self.inv_cov = np.linalg.inv(cov)
        return self

    def score(self, item):
        """Image-level anomaly score (max per-position Mahalanobis distance)."""
        if self.mean is None:
            raise RuntimeError("PaDiM is not fitted. Call .fit(...) first.")
        g, _ = self.extractor.grid_features([item])
        g = g[0, :, self.sel].numpy()
        diff = g - self.mean
        maha = np.einsum("pi,pij,pj->p", diff, self.inv_cov, diff)
        return float(np.sqrt(np.maximum(maha, 0)).max())


def zfuse(a, b, w=0.5):
    """Z-score normalize two score arrays and blend with weight ``w`` on ``a``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    za = (a - a.mean()) / (a.std() + 1e-9)
    zb = (b - b.mean()) / (b.std() + 1e-9)
    return w * za + (1 - w) * zb
