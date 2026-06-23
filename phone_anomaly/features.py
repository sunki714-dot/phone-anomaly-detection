"""Frozen ImageNet-pretrained feature extractor.

Mid-level feature maps (layer2 + layer3 for ResNet-family, or two dense blocks
for DenseNet) are pooled, aligned, and concatenated into per-patch descriptors.
The backbone is frozen — no training happens here.
"""

import torch
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image

# name -> (constructor, weights, default layers/indices, hook style)
_BACKBONES = {
    "resnet18": (
        tvm.resnet18,
        tvm.ResNet18_Weights.IMAGENET1K_V1,
        ["layer2", "layer3"],
        "layer",
    ),
    "wide_resnet50_2": (
        tvm.wide_resnet50_2,
        tvm.Wide_ResNet50_2_Weights.IMAGENET1K_V1,
        ["layer2", "layer3"],
        "layer",
    ),
    "densenet121": (
        tvm.densenet121,
        tvm.DenseNet121_Weights.IMAGENET1K_V1,
        [6, 8],
        "features",
    ),
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def available_backbones():
    """Return the list of supported backbone names."""
    return list(_BACKBONES)


class FeatureExtractor(torch.nn.Module):
    """Frozen backbone that turns images into per-patch feature descriptors.

    Parameters
    ----------
    backbone : str
        One of :func:`available_backbones`.
    layers : list, optional
        Override the default hook points.
    img_size : int
        Square resize applied before the backbone.
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(self, backbone="wide_resnet50_2", layers=None, img_size=256, device="cpu"):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(
                f"Unknown backbone {backbone!r}. Options: {available_backbones()}"
            )
        ctor, weights, default_layers, hook_type = _BACKBONES[backbone]

        self.backbone_name = backbone
        self.hook_type = hook_type
        self.layers = layers if layers is not None else default_layers
        self.img_size = img_size
        self.device = device

        self.body = ctor(weights=weights)
        self._feats = {}
        self._register_hooks()
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval().to(device)

        self.pool = torch.nn.AvgPool2d(3, 1, 1)
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )

    # -- hooks ------------------------------------------------------------
    def _hook_keys(self):
        if self.hook_type == "layer":
            return [str(layer) for layer in self.layers]
        return [str(idx) for idx in self.layers]

    def _register_hooks(self):
        def make_hook(name):
            def hook(_module, _inp, out):
                self._feats[name] = out

            return hook

        if self.hook_type == "layer":
            for layer in self.layers:
                getattr(self.body, str(layer)).register_forward_hook(make_hook(str(layer)))
        else:  # densenet: index into .features
            for idx in self.layers:
                self.body.features[idx].register_forward_hook(make_hook(str(idx)))

    # -- inputs -----------------------------------------------------------
    def _load_batch(self, items):
        """Accept a list of file paths and/or PIL images -> a normalized tensor."""
        tensors = []
        for item in items:
            img = item if isinstance(item, Image.Image) else Image.open(item)
            tensors.append(self.transform(img.convert("RGB")))
        return torch.stack(tensors).to(self.device)

    @torch.no_grad()
    def _forward_feats(self, x):
        """Run the backbone and fuse hooked feature maps -> (B, C, H, W)."""
        self._feats = {}
        _ = self.body(x)
        maps = [self.pool(self._feats[k]) for k in self._hook_keys()]
        base = maps[0].shape[-2:]
        maps = [
            F.interpolate(m, size=base, mode="bilinear", align_corners=False) for m in maps
        ]
        return torch.cat(maps, dim=1)

    # -- public API -------------------------------------------------------
    @torch.no_grad()
    def patch_features(self, items, batch_size=8):
        """Return ``(features, feature_map_size)``.

        ``features`` is ``(num_patches_total, channels)`` on CPU — every spatial
        location across the batch stacked into one matrix.
        """
        out = []
        fmap = None
        for i in range(0, len(items), batch_size):
            x = self._load_batch(items[i : i + batch_size])
            f = self._forward_feats(x)
            fmap = f.shape[-2:]
            _, C, _, _ = f.shape
            out.append(f.permute(0, 2, 3, 1).reshape(-1, C).cpu())
        return torch.cat(out, 0), fmap

    @torch.no_grad()
    def grid_features(self, items, batch_size=8):
        """Return ``(features, feature_map_size)`` keeping per-image structure.

        ``features`` is ``(num_images, num_patches_per_image, channels)`` — used
        by PaDiM, which models a Gaussian per spatial position.
        """
        out = []
        fmap = None
        for i in range(0, len(items), batch_size):
            x = self._load_batch(items[i : i + batch_size])
            f = self._forward_feats(x)
            fmap = f.shape[-2:]
            B, C, H, W = f.shape
            out.append(f.permute(0, 2, 3, 1).reshape(B, H * W, C).cpu())
        return torch.cat(out, 0), fmap
