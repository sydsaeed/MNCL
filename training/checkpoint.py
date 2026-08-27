from __future__ import annotations

import random
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class CheckpointState:
    epoch: int
    best_test_loss: float
    train_history: list[float]
    test_history: list[float]
    config: dict[str, Any] | None
    path: Path


def _config_to_dict(config: Any | None) -> dict[str, Any] | None:
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return vars(config).copy()


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_test_loss: float,
    train_history: list[float],
    test_history: list[float],
    config: Any | None = None,
) -> Path:
    """Save model, optimizer, history, and RNG state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_test_loss": float(best_test_loss),
        "train_history": [float(value) for value in train_history],
        "test_history": [float(value) for value in test_history],
        "config": _config_to_dict(config),
        "rng_state": _capture_rng_state(),
    }
    torch.save(payload, path)
    return path


def _torch_load(path: Path, device: torch.device | str) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
    restore_rng: bool = True,
) -> CheckpointState:
    """Load a checkpoint and restore trainable state."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = _torch_load(path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if restore_rng:
        _restore_rng_state(checkpoint.get("rng_state"))

    return CheckpointState(
        epoch=int(checkpoint.get("epoch", 0)),
        best_test_loss=float(checkpoint.get("best_test_loss", float("inf"))),
        train_history=[float(v) for v in checkpoint.get("train_history", [])],
        test_history=[float(v) for v in checkpoint.get("test_history", [])],
        config=checkpoint.get("config"),
        path=path,
    )
