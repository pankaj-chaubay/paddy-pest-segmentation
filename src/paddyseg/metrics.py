"""Segmentation metrics: IoU, pixel accuracy, precision/recall and COCO mAP.

* **IoU** (Jaccard index) is the overlap of a predicted and a true mask divided by their union.
* **Pixel accuracy** is the share of pixels whose class (insect or background) is right. It
  is easy to read but flattering when most of the image is background, so it is reported
  together with the foreground IoU.
* **Precision / recall** count a prediction as correct when it overlaps an unmatched true
  instance with IoU at or above a threshold (greedy matching by score).
* **mAP** is the COCO average precision over IoU thresholds 0.50:0.95, computed with
  ``pycocotools`` so the numbers match published benchmarks.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torch.utils.data import DataLoader

from paddyseg.dataset import CocoInstanceDataset, collate_fn
from paddyseg.engine import predict

BoolMasks = NDArray[np.bool_]
_COCO_STAT_NAMES = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


def mask_iou(pred: NDArray[Any], target: NDArray[Any]) -> float:
    """IoU of two binary masks of the same shape. Two empty masks count as a perfect match."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
    p, t = pred.astype(bool), target.astype(bool)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum() / union)


def pairwise_mask_iou(pred: BoolMasks, target: BoolMasks) -> NDArray[np.float64]:
    """IoU matrix of shape ``(len(pred), len(target))`` for stacks of binary masks."""
    if pred.shape[1:] != target.shape[1:]:
        raise ValueError(f"mask sizes differ: {pred.shape[1:]} vs {target.shape[1:]}")
    if len(pred) == 0 or len(target) == 0:
        return np.zeros((len(pred), len(target)))
    p = pred.reshape(len(pred), -1).astype(np.float64)
    t = target.reshape(len(target), -1).astype(np.float64)
    inter = p @ t.T
    union = p.sum(1)[:, None] + t.sum(1)[None, :] - inter
    result: NDArray[np.float64] = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return result


def pixel_accuracy(pred: NDArray[Any], target: NDArray[Any]) -> float:
    """Share of pixels where two label maps agree."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
    if pred.size == 0:
        raise ValueError("empty label map")
    return float((pred == target).mean())


@dataclass
class MatchCounts:
    """True positives, false positives, false negatives and the IoU of each true positive."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_ious: list[float] | None = None

    def __post_init__(self) -> None:
        if self.matched_ious is None:
            self.matched_ious = []

    def add(self, other: MatchCounts) -> None:
        """Accumulate another image's counts into this one."""
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        assert self.matched_ious is not None and other.matched_ious is not None
        self.matched_ious.extend(other.matched_ious)


def match_instances(
    pred_masks: BoolMasks,
    pred_scores: NDArray[Any],
    gt_masks: BoolMasks,
    iou_threshold: float = 0.5,
) -> MatchCounts:
    """Greedily match predictions (highest score first) to ground-truth instances."""
    ious = pairwise_mask_iou(pred_masks, gt_masks)
    order = np.argsort(-np.asarray(pred_scores))
    taken: set[int] = set()
    counts = MatchCounts()
    for i in order:
        best_j, best_iou = -1, iou_threshold
        for j in range(ious.shape[1]):
            if j not in taken and ious[i, j] >= best_iou:
                best_j, best_iou = j, float(ious[i, j])
        if best_j >= 0:
            taken.add(best_j)
            counts.tp += 1
            assert counts.matched_ious is not None
            counts.matched_ious.append(best_iou)
        else:
            counts.fp += 1
    counts.fn = gt_masks.shape[0] - len(taken)
    return counts


def coco_map(
    gt: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    iou_type: str = "segm",
    crowd_as_instances: bool = True,
) -> dict[str, float]:
    """COCO average precision for ``results`` against the ground truth ``gt``.

    Args:
        gt: Ground-truth COCO dictionary.
        results: Detections in COCO results format (``image_id``, ``category_id``, ``score``
            and ``segmentation`` as RLE, or ``bbox``).
        iou_type: ``"segm"`` for mask AP or ``"bbox"`` for box AP.
        crowd_as_instances: Treat ``iscrowd = 1`` annotations as normal instances. COCOeval
            otherwise ignores them, which would hide the CVAT brush-tool annotations.

    Returns:
        ``AP``, ``AP50``, ``AP75`` and AP for small, medium and large objects. Values are -1
        when a size bucket has no ground truth, following pycocotools.
    """
    if iou_type not in ("segm", "bbox"):
        raise ValueError("iou_type must be 'segm' or 'bbox'")
    annotations = []
    for ann in gt["annotations"]:
        ann = dict(ann)
        if crowd_as_instances:
            ann["iscrowd"] = 0
        annotations.append(ann)
    gt_dict = {"images": gt["images"], "categories": gt["categories"], "annotations": annotations}

    if not results:
        return {name: 0.0 for name in _COCO_STAT_NAMES}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(list(results))
        evaluator = COCOeval(coco_gt, coco_dt, iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {name: round(float(v), 4) for name, v in zip(_COCO_STAT_NAMES, evaluator.stats[:6], strict=True)}


def to_coco_results(
    output: Mapping[str, Tensor],
    image_id: int,
    label_to_category: Mapping[int, int],
    mask_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Convert one model output into COCO results records (RLE masks and xywh boxes)."""
    records: list[dict[str, Any]] = []
    masks = (output["masks"][:, 0] >= mask_threshold).numpy().astype(np.uint8)
    for k in range(masks.shape[0]):
        rle = mask_utils.encode(np.asfortranarray(masks[k]))
        rle["counts"] = rle["counts"].decode("ascii")
        x1, y1, x2, y2 = output["boxes"][k].tolist()
        records.append(
            {
                "image_id": image_id,
                "category_id": label_to_category[int(output["labels"][k])],
                "score": float(output["scores"][k]),
                "segmentation": rle,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            }
        )
    return records


@dataclass(frozen=True)
class SegmentationReport:
    """All evaluation numbers for one dataset split."""

    images: int
    instances: int
    predictions: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    foreground_iou: float
    pixel_accuracy: float
    mask_map: dict[str, float]
    box_map: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        return asdict(self)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: CocoInstanceDataset,
    device: torch.device,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    batch_size: int = 2,
) -> SegmentationReport:
    """Evaluate a model on every image of ``dataset`` (build it with evaluation transforms).

    Precision, recall, IoU and pixel accuracy use detections above ``score_threshold``.
    COCO mAP uses all detections, because AP integrates over the score ranking itself.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    counts = MatchCounts()
    correct_pixels = total_pixels = 0
    fg_inter = fg_union = 0
    coco_results: list[dict[str, Any]] = []
    n_instances = n_predictions = 0

    for images, targets in loader:
        outputs = predict(model, images, device, score_threshold=0.0)
        for out, target in zip(outputs, targets, strict=True):
            image_id = int(target["image_id"])
            coco_results.extend(to_coco_results(out, image_id, dataset.label_to_category, mask_threshold))

            keep = out["scores"] >= score_threshold
            pred = (out["masks"][keep][:, 0] >= mask_threshold).numpy()
            gt = target["masks"].numpy().astype(bool)
            counts.add(match_instances(pred, out["scores"][keep].numpy(), gt, iou_threshold))
            n_instances += gt.shape[0]
            n_predictions += pred.shape[0]

            height, width = gt.shape[-2:]
            pred_fg = pred.any(0) if pred.shape[0] else np.zeros((height, width), dtype=bool)
            gt_fg = gt.any(0) if gt.shape[0] else np.zeros((height, width), dtype=bool)
            correct_pixels += int((pred_fg == gt_fg).sum())
            total_pixels += gt_fg.size
            fg_inter += int(np.logical_and(pred_fg, gt_fg).sum())
            fg_union += int(np.logical_or(pred_fg, gt_fg).sum())

    precision = counts.tp / (counts.tp + counts.fp) if counts.tp + counts.fp else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if counts.tp + counts.fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    gt_coco = dataset.coco
    assert counts.matched_ious is not None
    return SegmentationReport(
        images=len(dataset),
        instances=n_instances,
        predictions=n_predictions,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        mean_matched_iou=round(float(np.mean(counts.matched_ious)) if counts.matched_ious else 0.0, 4),
        foreground_iou=round(fg_inter / fg_union, 4) if fg_union else 1.0,
        pixel_accuracy=round(correct_pixels / total_pixels, 4) if total_pixels else 0.0,
        mask_map=coco_map(gt_coco, coco_results, "segm", dataset.include_crowd),
        box_map=coco_map(gt_coco, coco_results, "bbox", dataset.include_crowd),
    )
