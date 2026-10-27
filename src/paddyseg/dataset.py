"""PyTorch dataset that serves COCO instance annotations in Mask R-CNN format."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import masks_to_boxes

from paddyseg.coco import annotation_to_mask, load_coco, validate_coco
from paddyseg.errors import DatasetError

LOGGER = logging.getLogger(__name__)

Target = dict[str, Tensor]
Sample = tuple[Tensor, Target]
TransformFn = Callable[[Any, Any], tuple[Any, Any]]


class CocoInstanceDataset(Dataset[Sample]):
    """Images and instance masks from a COCO file, ready for torchvision's Mask R-CNN.

    Each item is ``(image, target)``. ``image`` is a ``(3, H, W)`` tensor; ``target`` holds
    ``boxes`` (``float32``, ``x1, y1, x2, y2``), ``labels`` (``int64``, 1..K), ``masks``
    (``uint8``, ``N x H x W``), ``image_id``, ``area`` and ``iscrowd``.

    COCO category ids can be any integers; the dataset maps them to contiguous labels
    ``1..K`` because label 0 is reserved for the background. :attr:`label_to_category`
    gives the mapping back, which the evaluator needs.

    Args:
        annotations: Path to a COCO JSON file, or an already-loaded COCO dictionary.
        image_dir: Folder that holds the files named in ``images[*].file_name``.
        transforms: Optional callable from :func:`paddyseg.transforms.build_transforms`.
            Without it the image is returned as ``uint8``.
        include_crowd: Keep annotations marked ``iscrowd = 1``. CVAT marks shapes drawn with
            the brush tool as crowd even when they are single insects, so the default keeps them.
        min_mask_pixels: Instances with fewer pixels after augmentation are dropped.
    """

    def __init__(
        self,
        annotations: str | Path | Mapping[str, Any],
        image_dir: str | Path,
        transforms: TransformFn | None = None,
        include_crowd: bool = True,
        min_mask_pixels: int = 4,
    ) -> None:
        if isinstance(annotations, Mapping):
            validate_coco(annotations)
            self.coco: Mapping[str, Any] = annotations
        else:
            self.coco = load_coco(annotations)
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise DatasetError(f"Image folder not found: {self.image_dir}")
        if min_mask_pixels < 1:
            raise ValueError("min_mask_pixels must be at least 1")

        self.transforms = transforms
        self.include_crowd = include_crowd
        self.min_mask_pixels = min_mask_pixels

        categories = sorted(self.coco["categories"], key=lambda c: c["id"])
        self.category_to_label = {c["id"]: i + 1 for i, c in enumerate(categories)}
        self.label_to_category = {v: k for k, v in self.category_to_label.items()}
        self.class_names = ["__background__", *(c["name"] for c in categories)]

        self.images = sorted(self.coco["images"], key=lambda i: i["id"])
        self._anns_by_image: dict[int, list[Mapping[str, Any]]] = {i["id"]: [] for i in self.images}
        for ann in self.coco["annotations"]:
            if include_crowd or not ann.get("iscrowd", 0):
                self._anns_by_image[ann["image_id"]].append(ann)

        missing = [i["file_name"] for i in self.images if not (self.image_dir / i["file_name"]).is_file()]
        if missing:
            raise DatasetError(
                f"{len(missing)} image(s) listed in the annotations are missing from "
                f"{self.image_dir}, for example {missing[:3]}"
            )

    @property
    def num_classes(self) -> int:
        """Number of classes including the background."""
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Sample:
        info = self.images[index]
        image = self._read_image(info)
        height, width = int(info["height"]), int(info["width"])
        anns = self._anns_by_image[info["id"]]

        if anns:
            mask_array = np.stack([annotation_to_mask(a, height, width) for a in anns])
        else:
            mask_array = np.zeros((0, height, width), dtype=np.uint8)
        masks = tv_tensors.Mask(torch.from_numpy(mask_array))
        labels = torch.tensor([self.category_to_label[a["category_id"]] for a in anns], dtype=torch.int64)
        iscrowd = torch.tensor([int(a.get("iscrowd", 0)) for a in anns], dtype=torch.int64)

        img: Any = tv_tensors.Image(image)
        target: dict[str, Any] = {"masks": masks, "labels": labels, "iscrowd": iscrowd}
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return torch.as_tensor(img), self._finalise_target(target, int(info["id"]))

    def _read_image(self, info: Mapping[str, Any]) -> Tensor:
        path = self.image_dir / info["file_name"]
        try:
            image = read_rgb(path)
        except (UnidentifiedImageError, OSError) as exc:
            raise DatasetError(f"Cannot read image {path}: {exc}") from exc
        expected = (int(info["height"]), int(info["width"]))
        if tuple(image.shape[-2:]) != expected:
            raise DatasetError(f"{path} is {tuple(image.shape[-2:])} (H, W) but the annotation says {expected}")
        return image

    def _finalise_target(self, target: Mapping[str, Any], image_id: int) -> Target:
        """Drop instances that augmentation pushed out of frame and recompute tight boxes."""
        masks = torch.as_tensor(target["masks"]).to(torch.uint8)
        pixel_counts = masks.flatten(1).sum(dim=1) if masks.numel() else torch.zeros(0)
        keep = pixel_counts >= self.min_mask_pixels
        masks = masks[keep]
        labels = torch.as_tensor(target["labels"])[keep]
        iscrowd = torch.as_tensor(target["iscrowd"])[keep]

        if masks.shape[0]:
            boxes = masks_to_boxes(masks)
            boxes[:, 2:] += 1.0  # masks_to_boxes returns inclusive max coordinates
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        return {
            "boxes": boxes.to(torch.float32),
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": masks.flatten(1).sum(dim=1).to(torch.float32) if masks.shape[0] else torch.zeros(0),
            "iscrowd": iscrowd,
        }


def read_rgb(path: str | Path) -> Tensor:
    """Read an image file as a ``uint8`` tensor of shape ``(3, H, W)`` in RGB order."""
    with Image.open(path) as img:
        array = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


def collate_fn(batch: Sequence[Sample]) -> tuple[tuple[Tensor, ...], tuple[Target, ...]]:
    """Keep images of different sizes apart; detection models take lists, not stacked tensors."""
    images, targets = zip(*batch, strict=True)
    return tuple(images), tuple(targets)
