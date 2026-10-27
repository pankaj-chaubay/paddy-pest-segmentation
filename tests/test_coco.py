from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paddyseg.coco import annotation_to_mask, load_coco, split_coco, summarize, validate_coco
from paddyseg.errors import AnnotationError

REAL_ANNOTATIONS = Path(__file__).resolve().parents[1] / "data" / "annotations" / "paddy_pest_coco.json"


def test_load_valid_file(synthetic: tuple[Path, Path]) -> None:
    coco = load_coco(synthetic[0])
    assert len(coco["images"]) == 6


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AnnotationError, match="not found"):
        load_coco(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(AnnotationError, match="not valid JSON"):
        load_coco(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.pop("categories"), "categories"),
        (lambda c: c["annotations"][0].update(image_id=999), "unknown image"),
        (lambda c: c["annotations"][0].update(category_id=999), "unknown category"),
        (lambda c: c["images"].append(dict(c["images"][0])), "duplicate image"),
        (lambda c: c["images"][0].update(width=0), "width/height"),
        (lambda c: c["annotations"][0].update(segmentation=[]), "no segmentation"),
    ],
)
def test_validation_rules(synthetic: tuple[Path, Path], mutate, message: str) -> None:  # type: ignore[no-untyped-def]
    coco = json.loads(synthetic[0].read_text())
    mutate(coco)
    with pytest.raises(AnnotationError, match=message):
        validate_coco(coco)


def test_polygon_and_rle_masks_match_area(synthetic: tuple[Path, Path]) -> None:
    coco = load_coco(synthetic[0])
    images = {i["id"]: i for i in coco["images"]}
    formats = set()
    for ann in coco["annotations"]:
        info = images[ann["image_id"]]
        mask = annotation_to_mask(ann, info["height"], info["width"])
        assert mask.dtype == np.uint8 and mask.shape == (info["height"], info["width"])
        assert abs(int(mask.sum()) - ann["area"]) <= 1
        formats.add(type(ann["segmentation"]).__name__)
    assert formats == {"list", "dict"}


def test_rle_size_mismatch_raises() -> None:
    ann = {"id": 1, "segmentation": {"counts": [4], "size": [2, 2]}}
    with pytest.raises(AnnotationError, match="does not match"):
        annotation_to_mask(ann, 3, 3)


def test_degenerate_polygon_raises() -> None:
    with pytest.raises(AnnotationError, match="at least 3 points"):
        annotation_to_mask({"id": 1, "segmentation": [[1, 1, 2, 2]]}, 5, 5)


def test_split_is_disjoint_complete_and_deterministic(synthetic: tuple[Path, Path]) -> None:
    coco = load_coco(synthetic[0])
    first = split_coco(coco, (0.5, 0.25, 0.25), seed=1)
    second = split_coco(coco, (0.5, 0.25, 0.25), seed=1)
    ids = {name: {i["id"] for i in s["images"]} for name, s in first.items()}
    assert ids == {name: {i["id"] for i in s["images"]} for name, s in second.items()}
    assert set().union(*ids.values()) == {i["id"] for i in coco["images"]}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]
    for subset in first.values():
        image_ids = {i["id"] for i in subset["images"]}
        assert all(a["image_id"] in image_ids for a in subset["annotations"])


def test_split_rejects_bad_ratios(synthetic: tuple[Path, Path]) -> None:
    coco = load_coco(synthetic[0])
    with pytest.raises(ValueError):
        split_coco(coco, (0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="empty"):
        split_coco(coco, (0.98, 0.01, 0.01))


def test_summary(synthetic: tuple[Path, Path]) -> None:
    summary = summarize(load_coco(synthetic[0]))
    assert summary.images == 6
    assert summary.images_without_annotations == 1
    assert summary.rle_annotations == 1 and summary.crowd_annotations == 1
    assert sum(summary.size_buckets.values()) == summary.annotations


@pytest.mark.skipif(not REAL_ANNOTATIONS.is_file(), reason="annotation file not present")
def test_real_cvat_export_is_valid() -> None:
    coco = load_coco(REAL_ANNOTATIONS)
    summary = summarize(coco)
    assert summary.images == 112 and summary.annotations == 127
    for ann in coco["annotations"][:20]:
        info = next(i for i in coco["images"] if i["id"] == ann["image_id"])
        assert annotation_to_mask(ann, info["height"], info["width"]).any()
