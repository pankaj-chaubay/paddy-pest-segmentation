"""Shared fixtures: a tiny synthetic COCO dataset written to a temporary folder.

The real paddy images are not redistributed, so the tests build their own: coloured
backgrounds with bright elliptical "insects" whose polygons are stored in COCO format.
One annotation uses uncompressed RLE (like CVAT's brush tool) and one image has no
annotations, so both edge cases are always covered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

SIZE = (96, 128)  # (height, width)


def _ellipse_polygon(cx: float, cy: float, rx: float, ry: float, points: int = 24) -> list[float]:
    angles = np.linspace(0, 2 * np.pi, points, endpoint=False)
    xy = np.stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)], axis=1)
    return [round(float(v), 2) for v in xy.reshape(-1)]


def _uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    flat = mask.flatten(order="F")
    counts, current, run = [], 0, 0
    for value in flat:
        if value == current:
            run += 1
        else:
            counts.append(run)
            current, run = value, 1
    counts.append(run)
    return {"counts": counts, "size": list(mask.shape)}


def make_dataset(root: Path, n_images: int = 6) -> tuple[Path, Path]:
    """Write ``n_images`` images and a COCO file; return (annotation path, image folder)."""
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    height, width = SIZE
    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 7, "name": "insect", "supercategory": "pest"}],
    }
    ann_id = 1
    for image_id in range(1, n_images + 1):
        pixels = np.full((height, width, 3), (40, 90, 40), dtype=np.uint8)
        name = f"img_{image_id:03d}.png"
        coco["images"].append({"id": image_id, "file_name": name, "height": height, "width": width})
        n_objects = 0 if image_id == n_images else 1 + image_id % 2
        for _ in range(n_objects):
            cx, cy = rng.uniform(30, width - 30), rng.uniform(25, height - 25)
            poly = _ellipse_polygon(cx, cy, rng.uniform(8, 18), rng.uniform(6, 12))
            rle = mask_utils.merge(mask_utils.frPyObjects([poly], height, width))
            mask = mask_utils.decode(rle).astype(bool)
            pixels[mask] = (230, 200, 60)
            use_rle = ann_id == 2
            x, y, w, h = mask_utils.toBbox(rle).tolist()
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 7,
                    "segmentation": _uncompressed_rle(mask.astype(np.uint8)) if use_rle else [poly],
                    "area": float(mask.sum()),
                    "bbox": [x, y, w, h],
                    "iscrowd": 1 if use_rle else 0,
                }
            )
            ann_id += 1
        Image.fromarray(pixels).save(image_dir / name)
    ann_path = root / "annotations.json"
    ann_path.write_text(json.dumps(coco), encoding="utf-8")
    return ann_path, image_dir


@pytest.fixture()
def synthetic(tmp_path: Path) -> tuple[Path, Path]:
    return make_dataset(tmp_path)
