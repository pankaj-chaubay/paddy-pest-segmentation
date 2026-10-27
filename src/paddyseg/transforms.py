"""Augmentation pipelines that keep images and instance masks aligned.

The pipelines use ``torchvision.transforms.v2``. Images are wrapped as
:class:`torchvision.tv_tensors.Image` and masks as :class:`torchvision.tv_tensors.Mask`, so
every geometric transform (flip, scale, rotation) is applied to both with the same random
parameters, and masks are resampled with nearest-neighbour interpolation. Photometric
transforms (brightness, contrast) change only the image.

Bounding boxes are not transformed here. After rotation a transformed box is a loose fit, so
:mod:`paddyseg.dataset` recomputes tight boxes from the transformed masks instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.transforms import v2


@dataclass(frozen=True)
class AugmentConfig:
    """Strength of each augmentation used during training.

    Attributes:
        hflip_p: Probability of a horizontal flip.
        vflip_p: Probability of a vertical flip.
        rotation_degrees: Maximum absolute rotation in degrees.
        scale: Range of random zoom factors (``1.0`` keeps the size).
        brightness: Maximum relative brightness change.
        contrast: Maximum relative contrast change.
    """

    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rotation_degrees: float = 15.0
    scale: tuple[float, float] = (0.8, 1.2)
    brightness: float = 0.2
    contrast: float = 0.2

    def __post_init__(self) -> None:
        for name in ("hflip_p", "vflip_p"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.rotation_degrees < 0:
            raise ValueError("rotation_degrees must be non-negative")
        low, high = self.scale
        if not 0 < low <= high:
            raise ValueError(f"scale must satisfy 0 < low <= high, got {self.scale}")
        if self.brightness < 0 or self.contrast < 0:
            raise ValueError("brightness and contrast must be non-negative")


def build_transforms(train: bool, config: AugmentConfig | None = None) -> v2.Compose:
    """Build the transform pipeline for training or evaluation.

    Args:
        train: ``True`` adds random augmentation; ``False`` only converts the image to float.
        config: Augmentation strengths. Defaults to :class:`AugmentConfig`.

    Returns:
        A callable that takes ``(image, target)`` and returns them transformed. The image
        comes out as ``float32`` in ``[0, 1]``.
    """
    to_float = v2.ToDtype(torch.float32, scale=True)
    if not train:
        return v2.Compose([to_float])

    cfg = config or AugmentConfig()
    steps: list[v2.Transform] = [
        v2.RandomHorizontalFlip(p=cfg.hflip_p),
        v2.RandomVerticalFlip(p=cfg.vflip_p),
    ]
    if cfg.rotation_degrees > 0 or cfg.scale != (1.0, 1.0):
        steps.append(v2.RandomAffine(degrees=cfg.rotation_degrees, scale=cfg.scale))
    if cfg.brightness > 0 or cfg.contrast > 0:
        steps.append(v2.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast))
    steps.append(to_float)
    return v2.Compose(steps)
