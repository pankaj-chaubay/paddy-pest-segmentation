"""Command-line interface: ``paddyseg <command> [options]``.

Commands
--------
``inspect``   Summarise an annotation file (counts, formats, object sizes).
``split``     Write train/val/test annotation files, split by image.
``augment``   Save before/after pictures that show masks follow each augmentation.
``train``     Fine-tune Mask R-CNN and save checkpoints plus a loss history.
``evaluate``  Report IoU, pixel accuracy, precision/recall and COCO mAP for a checkpoint.
``predict``   Draw predicted masks on one or more images.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from paddyseg.coco import load_coco, split_coco, summarize, write_coco
from paddyseg.dataset import CocoInstanceDataset, collate_fn, read_rgb
from paddyseg.engine import TrainConfig, fit, load_checkpoint, predict
from paddyseg.errors import PaddySegError
from paddyseg.metrics import evaluate
from paddyseg.model import build_model
from paddyseg.transforms import AugmentConfig, build_transforms
from paddyseg.visualize import overlay, save_grid

LOGGER = logging.getLogger("paddyseg")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def cmd_inspect(args: argparse.Namespace) -> int:
    summary = summarize(load_coco(args.annotations))
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    splits = split_coco(load_coco(args.annotations), ratios=args.ratios, seed=args.seed)
    for name, subset in splits.items():
        path = write_coco(subset, Path(args.out_dir) / f"{name}.json")
        LOGGER.info(
            "%-5s %4d images %4d annotations -> %s", name, len(subset["images"]), len(subset["annotations"]), path
        )
    return 0


def cmd_augment(args: argparse.Namespace) -> int:
    plain = CocoInstanceDataset(args.annotations, args.images, transforms=build_transforms(train=False))
    augmented = CocoInstanceDataset(args.annotations, args.images, transforms=build_transforms(train=True))
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(plain))[: args.count].tolist()
    panels, titles = [], []
    for i in indices:
        for name, ds in (("original", plain), ("augmented", augmented)):
            image, target = ds[i]
            panels.append(overlay(image, target["masks"], target["boxes"]))
            titles.append(f"{ds.images[i]['file_name']} - {name}")
    path = save_grid(panels, titles, args.out, cols=2)
    LOGGER.info("Saved %s", path)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    device = _device(args.device)
    aug = AugmentConfig(rotation_degrees=args.rotation, scale=(args.scale_min, args.scale_max))
    train_ds = CocoInstanceDataset(args.train, args.images, transforms=build_transforms(True, aug))
    val_ds = CocoInstanceDataset(args.val, args.images, transforms=build_transforms(False)) if args.val else None
    loader_kw = {"batch_size": args.batch_size, "collate_fn": collate_fn, "num_workers": args.workers}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw) if val_ds is not None else None

    model = build_model(
        train_ds.num_classes, pretrained=not args.no_pretrained, min_size=args.min_size, max_size=args.max_size
    )
    config = TrainConfig(epochs=args.epochs, lr=args.lr, lr_step=args.lr_step, seed=args.seed)
    LOGGER.info(
        "Training on %s: %d train images, %s val images, %d classes",
        device,
        len(train_ds),
        len(val_ds) if val_ds else 0,
        train_ds.num_classes - 1,
    )
    fit(model, train_loader, val_loader, config, args.out, device)
    meta = {
        "class_names": train_ds.class_names,
        "label_to_category": train_ds.label_to_category,
        "min_size": args.min_size,
        "max_size": args.max_size,
    }
    (Path(args.out) / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


def _load_trained(checkpoint: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    meta_path = Path(checkpoint).with_name("model_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    model = build_model(
        num_classes, pretrained=False, min_size=meta.get("min_size", 800), max_size=meta.get("max_size", 1333)
    )
    return load_checkpoint(model, checkpoint, device)


def cmd_evaluate(args: argparse.Namespace) -> int:
    device = _device(args.device)
    dataset = CocoInstanceDataset(args.annotations, args.images, transforms=build_transforms(False))
    model = _load_trained(args.checkpoint, dataset.num_classes, device)
    report = evaluate(model, dataset, device, score_threshold=args.score_threshold)
    text = json.dumps(report.to_dict(), indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    device = _device(args.device)
    model = _load_trained(args.checkpoint, args.num_classes, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = read_rgb(path).float() / 255
        result = predict(model, [image], device, args.score_threshold)[0]
        captions = [f"{s:.2f}" for s in result["scores"].tolist()]
        panel = overlay(image, result["masks"], result["boxes"], captions)
        target = save_grid(
            [panel], [f"{Path(path).name}: {len(captions)} insects"], out_dir / f"{Path(path).stem}_pred.png", cols=1
        )
        LOGGER.info("%s -> %s", path, target)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddyseg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="summarise an annotation file")
    p.add_argument("annotations")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("split", help="split by image into train/val/test files")
    p.add_argument("annotations")
    p.add_argument("--out-dir", default="data/splits")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("augment", help="save before/after augmentation pictures")
    p.add_argument("annotations")
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="outputs/augmentation_preview.png")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_augment)

    p = sub.add_parser("train", help="fine-tune Mask R-CNN")
    p.add_argument("--train", required=True, help="train annotation file")
    p.add_argument("--val", help="validation annotation file")
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="outputs/run")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--lr-step", type=int, default=3)
    p.add_argument("--rotation", type=float, default=15.0)
    p.add_argument("--scale-min", type=float, default=0.8)
    p.add_argument("--scale-max", type=float, default=1.2)
    p.add_argument("--min-size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-pretrained", action="store_true", help="start from random weights")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p.add_argument("annotations")
    p.add_argument("--images", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--out", help="also write the report to this JSON file")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="draw predictions on images")
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-classes", type=int, default=2, help="including background")
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--out-dir", default="outputs/predictions")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code instead of raising for expected errors."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return int(args.func(args))
    except (PaddySegError, FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
