"""End-to-end smoke tests on CPU with randomly initialised weights and small images."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from paddyseg.cli import main
from paddyseg.dataset import CocoInstanceDataset, collate_fn
from paddyseg.engine import TrainConfig, fit, load_checkpoint, predict
from paddyseg.metrics import evaluate
from paddyseg.model import build_model
from paddyseg.transforms import build_transforms

CPU = torch.device("cpu")


def _tiny_model(num_classes: int = 2) -> torch.nn.Module:
    return build_model(num_classes, pretrained=False, min_size=96, max_size=128)


def test_build_model_heads() -> None:
    model = _tiny_model(4)
    assert model.roi_heads.box_predictor.cls_score.out_features == 4
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 4
    with pytest.raises(ValueError):
        build_model(1, pretrained=False)


def test_fit_evaluate_and_reload(synthetic: tuple[Path, Path], tmp_path: Path) -> None:
    torch.manual_seed(0)
    train = CocoInstanceDataset(*synthetic, transforms=build_transforms(True))
    val = CocoInstanceDataset(*synthetic, transforms=build_transforms(False))
    loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val, batch_size=2, collate_fn=collate_fn)

    model = _tiny_model()
    history = fit(model, loader, val_loader, TrainConfig(epochs=1, lr=0.001), tmp_path / "run", CPU)
    assert len(history) == 1 and history[0].val_loss is not None
    assert {"last.pt", "best.pt", "history.csv"} <= {p.name for p in (tmp_path / "run").iterdir()}
    with (tmp_path / "run" / "history.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 1

    reloaded = load_checkpoint(_tiny_model(), tmp_path / "run" / "best.pt", CPU)
    outputs = predict(reloaded, [val[0][0]], CPU, score_threshold=0.0)
    assert set(outputs[0]) == {"boxes", "labels", "scores", "masks"}

    report = evaluate(reloaded, val, CPU)
    assert report.images == len(val) and 0.0 <= report.pixel_accuracy <= 1.0
    assert set(report.mask_map) == {"AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"}


def test_cli_inspect_and_split(
    synthetic: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect", str(synthetic[0])]) == 0
    assert '"images": 6' in capsys.readouterr().out
    assert main(["split", str(synthetic[0]), "--out-dir", str(tmp_path / "s"), "--ratios", "0.5", "0.25", "0.25"]) == 0
    assert sorted(p.name for p in (tmp_path / "s").iterdir()) == ["test.json", "train.json", "val.json"]


def test_cli_reports_errors(tmp_path: Path) -> None:
    assert main(["inspect", str(tmp_path / "missing.json")]) == 2
