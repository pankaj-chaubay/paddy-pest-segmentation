from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from paddyseg.coco import load_coco
from paddyseg.dataset import CocoInstanceDataset
from paddyseg.metrics import coco_map, mask_iou, match_instances, pairwise_mask_iou, pixel_accuracy, to_coco_results


def _square(top: int, left: int, size: int = 4, shape: tuple[int, int] = (10, 10)) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[top : top + size, left : left + size] = True
    return m


def test_mask_iou_cases() -> None:
    a = _square(0, 0)
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, _square(5, 5)) == 0.0
    assert mask_iou(a, _square(0, 2)) == pytest.approx(8 / 24)
    assert mask_iou(np.zeros((3, 3)), np.zeros((3, 3))) == 1.0
    with pytest.raises(ValueError):
        mask_iou(a, np.zeros((2, 2)))


def test_pairwise_matches_single() -> None:
    preds = np.stack([_square(0, 0), _square(0, 2)])
    gts = np.stack([_square(0, 0), _square(5, 5)])
    ious = pairwise_mask_iou(preds, gts)
    assert ious.shape == (2, 2)
    assert ious[0, 0] == 1.0 and ious[1, 0] == pytest.approx(8 / 24) and ious[0, 1] == 0.0


def test_pixel_accuracy() -> None:
    assert pixel_accuracy(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])) == 0.75


def test_match_instances_greedy_by_score() -> None:
    gt = np.stack([_square(0, 0)])
    preds = np.stack([_square(0, 1), _square(0, 0)])
    counts = match_instances(preds, np.array([0.4, 0.9]), gt)
    assert (counts.tp, counts.fp, counts.fn) == (1, 1, 0)
    assert counts.matched_ious == [1.0]  # the higher-scored, perfect mask wins


def test_match_instances_no_predictions() -> None:
    counts = match_instances(np.zeros((0, 10, 10), bool), np.zeros(0), np.stack([_square(0, 0)]))
    assert (counts.tp, counts.fp, counts.fn) == (0, 0, 1)


def test_perfect_predictions_score_full_map(synthetic: tuple[Path, Path]) -> None:
    ds = CocoInstanceDataset(*synthetic)
    results = []
    for i in range(len(ds)):
        _, target = ds[i]
        output = {
            "boxes": target["boxes"],
            "labels": target["labels"],
            "scores": torch.ones(len(target["labels"])),
            "masks": target["masks"][:, None].float(),
        }
        results.extend(to_coco_results(output, int(target["image_id"]), ds.label_to_category))
    gt = load_coco(synthetic[0])
    assert coco_map(gt, results, "segm")["AP"] == pytest.approx(1.0)
    assert coco_map(gt, results, "bbox")["AP50"] == pytest.approx(1.0)


def test_empty_results_score_zero(synthetic: tuple[Path, Path]) -> None:
    assert coco_map(load_coco(synthetic[0]), [], "segm")["AP"] == 0.0
