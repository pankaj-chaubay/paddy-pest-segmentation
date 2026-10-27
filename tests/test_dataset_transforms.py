from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from paddyseg.dataset import CocoInstanceDataset, collate_fn
from paddyseg.errors import DatasetError
from paddyseg.transforms import AugmentConfig, build_transforms


def test_item_format(synthetic: tuple[Path, Path]) -> None:
    ds = CocoInstanceDataset(*synthetic, transforms=build_transforms(train=False))
    image, target = ds[0]
    assert image.dtype == torch.float32 and image.shape == (3, 96, 128)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
    n = target["masks"].shape[0]
    assert target["boxes"].shape == (n, 4) and target["labels"].shape == (n,)
    assert target["masks"].dtype == torch.uint8
    assert set(target["labels"].tolist()) <= {1}  # COCO id 7 is mapped to label 1
    assert ds.label_to_category == {1: 7} and ds.num_classes == 2


def test_boxes_are_tight_around_masks(synthetic: tuple[Path, Path]) -> None:
    ds = CocoInstanceDataset(*synthetic)
    _, target = ds[0]
    for box, mask in zip(target["boxes"], target["masks"], strict=True):
        ys, xs = torch.nonzero(mask, as_tuple=True)
        assert box.tolist() == [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1]


def test_image_without_annotations(synthetic: tuple[Path, Path]) -> None:
    ds = CocoInstanceDataset(*synthetic)
    _, target = ds[len(ds) - 1]
    assert target["boxes"].shape == (0, 4) and target["masks"].shape[0] == 0


def test_exclude_crowd(synthetic: tuple[Path, Path]) -> None:
    with_crowd = CocoInstanceDataset(*synthetic)
    without = CocoInstanceDataset(*synthetic, include_crowd=False)
    count = lambda d: sum(d[i][1]["labels"].numel() for i in range(len(d)))  # noqa: E731
    assert count(with_crowd) == count(without) + 1


@pytest.mark.parametrize("seed", range(5))
def test_augmentation_moves_image_and_mask_together(synthetic: tuple[Path, Path], seed: int) -> None:
    """Insects are the only bright-yellow pixels, so after any augmentation the yellow pixels
    in the image must still sit under the masks."""
    cfg = AugmentConfig(hflip_p=0.5, vflip_p=0.5, rotation_degrees=30, scale=(0.7, 1.3), brightness=0.0, contrast=0.0)
    ds = CocoInstanceDataset(*synthetic, transforms=build_transforms(True, cfg))
    torch.manual_seed(seed)
    image, target = ds[1]
    if target["masks"].shape[0] == 0:
        pytest.skip("augmentation moved every insect out of frame")
    insect_pixels = (image[0] > 0.6) & (image[2] < 0.5)
    union = target["masks"].bool().any(0)
    agreement = (insect_pixels == union).float().mean().item()
    assert agreement > 0.98


def test_missing_image_raises(synthetic: tuple[Path, Path]) -> None:
    ann_path, image_dir = synthetic
    next(image_dir.iterdir()).unlink()
    with pytest.raises(DatasetError, match="missing"):
        CocoInstanceDataset(ann_path, image_dir)


def test_size_mismatch_raises(synthetic: tuple[Path, Path]) -> None:
    ann_path, image_dir = synthetic
    coco = json.loads(ann_path.read_text())
    coco["images"][0]["width"] = 999
    ds = CocoInstanceDataset(coco, image_dir)
    with pytest.raises(DatasetError, match="annotation says"):
        ds[0]


def test_bad_augment_config() -> None:
    with pytest.raises(ValueError):
        AugmentConfig(hflip_p=1.5)
    with pytest.raises(ValueError):
        AugmentConfig(scale=(1.2, 0.8))


def test_collate_keeps_lists(synthetic: tuple[Path, Path]) -> None:
    ds = CocoInstanceDataset(*synthetic, transforms=build_transforms(False))
    images, targets = collate_fn([ds[0], ds[1]])
    assert len(images) == 2 and len(targets) == 2
