"""Training, validation and inference loops."""

from __future__ import annotations

import csv
import logging
import math
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from paddyseg.dataset import Target

LOGGER = logging.getLogger(__name__)
Batch = tuple[Sequence[Tensor], Sequence[Target]]


@dataclass(frozen=True)
class TrainConfig:
    """Optimiser and schedule settings.

    The defaults follow the coursework: SGD with momentum, learning rate 0.005 that drops by
    10x every ``lr_step`` epochs, and a small batch because Mask R-CNN is memory-hungry.
    """

    epochs: int = 10
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_step: int = 3
    lr_gamma: float = 0.1
    grad_clip: float | None = 10.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.lr_step < 1:
            raise ValueError("lr_step must be at least 1")


@dataclass(frozen=True)
class EpochResult:
    """Losses recorded for one epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None
    lr: float
    seconds: float


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so runs can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_device(batch: Batch, device: torch.device) -> tuple[list[Tensor], list[Target]]:
    images, targets = batch
    return (
        [img.to(device) for img in images],
        [{k: v.to(device) for k, v in t.items()} for t in targets],
    )


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = 10.0,
) -> dict[str, float]:
    """Run one pass over ``loader`` and update the weights.

    Returns:
        Mean of each loss component and of the total (key ``"loss"``).

    Raises:
        FloatingPointError: If the loss becomes NaN or infinite, which usually means the
            learning rate is too high or an annotation is broken.
    """
    model.train()
    sums: dict[str, float] = {}
    steps = 0
    for batch in loader:
        images, targets = _to_device(batch, device)
        loss_dict: dict[str, Tensor] = model(images, targets)
        loss = torch.stack(list(loss_dict.values())).sum()
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"Loss is {loss.item()} at step {steps}: {_floats(loss_dict)}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key, value in {**_floats(loss_dict), "loss": loss.item()}.items():
            sums[key] = sums.get(key, 0.0) + value
        steps += 1
    if steps == 0:
        raise ValueError("The training loader produced no batches")
    return {k: v / steps for k, v in sums.items()}


@torch.no_grad()
def validation_loss(model: nn.Module, loader: Iterable[Batch], device: torch.device) -> float:
    """Mean total loss on a held-out loader, without updating weights.

    torchvision detection models only return losses in training mode. Their batch-norm
    layers are frozen and they have no dropout, so training mode under ``no_grad`` measures
    the same loss without side effects.
    """
    model.train()
    total, steps = 0.0, 0
    for batch in loader:
        images, targets = _to_device(batch, device)
        loss_dict: dict[str, Tensor] = model(images, targets)
        total += float(torch.stack(list(loss_dict.values())).sum().item())
        steps += 1
    if steps == 0:
        raise ValueError("The validation loader produced no batches")
    return total / steps


@torch.no_grad()
def predict(
    model: nn.Module,
    images: Sequence[Tensor],
    device: torch.device,
    score_threshold: float = 0.5,
) -> list[dict[str, Tensor]]:
    """Run inference and drop detections below ``score_threshold``.

    Returns:
        One dictionary per image with ``boxes``, ``labels``, ``scores`` and ``masks``
        (``float`` probabilities, ``N x 1 x H x W``), all on the CPU.
    """
    model.eval()
    outputs: list[dict[str, Tensor]] = model([img.to(device) for img in images])
    results = []
    for out in outputs:
        keep = out["scores"] >= score_threshold
        results.append({k: v[keep].cpu() for k, v in out.items()})
    return results


def fit(
    model: nn.Module,
    train_loader: Iterable[Batch],
    val_loader: Iterable[Batch] | None,
    config: TrainConfig,
    output_dir: str | Path,
    device: torch.device,
) -> list[EpochResult]:
    """Train for ``config.epochs`` epochs, keeping the checkpoint with the lowest validation loss.

    Files written to ``output_dir``: ``last.pt``, ``best.pt`` (only with a validation
    loader) and ``history.csv`` with one row per epoch.

    Returns:
        The per-epoch results, in order.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)

    history: list[EpochResult] = []
    best = math.inf
    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_stats = train_one_epoch(model, train_loader, optimizer, device, config.grad_clip)
        val = validation_loss(model, val_loader, device) if val_loader is not None else None
        scheduler.step()

        result = EpochResult(epoch, train_stats["loss"], val, lr, time.perf_counter() - start)
        history.append(result)
        LOGGER.info(
            "epoch %d/%d  train_loss=%.4f  val_loss=%s  lr=%.5f  (%.0fs)",
            epoch,
            config.epochs,
            result.train_loss,
            "n/a" if val is None else f"{val:.4f}",
            lr,
            result.seconds,
        )

        state: dict[str, Any] = {"model": model.state_dict(), "epoch": epoch, "config": asdict(config)}
        torch.save(state, out / "last.pt")
        if val is not None and val < best:
            best = val
            torch.save(state, out / "best.pt")
        _write_history(history, out / "history.csv")
    return history


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> nn.Module:
    """Load weights saved by :func:`fit` into ``model``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model", state))
    return model.to(device)


def _floats(losses: dict[str, Tensor]) -> dict[str, float]:
    return {k: float(v.item()) for k, v in losses.items()}


def _write_history(history: Sequence[EpochResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(history[0])))
        writer.writeheader()
        writer.writerows(asdict(h) for h in history)
