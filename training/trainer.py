from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from datasets.bpr_dataset import build_bpr_dataloader
from training.checkpoint import save_checkpoint


@dataclass
class EpochLossStats:
    total_loss: float
    bpr_loss: float
    contrastive_loss: float
    l2_loss: float


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve auto, CUDA, MPS, or CPU device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def move_graphs_to_device(graphs: Any, device: torch.device) -> Any:
    """Move all graph views to the selected device."""
    graphs.user_item = graphs.user_item.to(device)
    graphs.kg = graphs.kg.to(device)
    graphs.user_item_entity = graphs.user_item_entity.to(device)
    return graphs


def _pair_scores(
    user_embeddings: Tensor,
    item_embeddings: Tensor,
    user_ids: Tensor,
    item_ids: Tensor,
) -> Tensor:
    users = user_embeddings[user_ids]
    items = item_embeddings[item_ids]
    return (users * items).sum(dim=-1)


def _batch_to_device(
    batch: tuple[Tensor, Tensor, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    return tuple(t.to(device, non_blocking=True) for t in batch)


def _contrastive_ids(
    user_ids: Tensor,
    positive_item_ids: Tensor,
    negative_item_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    unique_users = torch.unique(user_ids)
    unique_items = torch.unique(
        torch.cat([positive_item_ids, negative_item_ids], dim=0)
    )
    return unique_users, unique_items


def _empty_running() -> dict[str, float]:
    return {
        "total": 0.0,
        "bpr": 0.0,
        "contrastive": 0.0,
        "l2": 0.0,
        "samples": 0.0,
    }


def _update_running(running: dict[str, float], loss_output, batch_size: int) -> None:
    running["total"] += float(loss_output.total_loss.detach()) * batch_size
    running["bpr"] += float(loss_output.bpr_loss.detach()) * batch_size
    running["contrastive"] += float(loss_output.contrastive_loss.detach()) * batch_size
    running["l2"] += float(loss_output.l2_loss.detach()) * batch_size
    running["samples"] += batch_size


def _finalize_running(running: dict[str, float]) -> EpochLossStats:
    samples = max(running["samples"], 1.0)
    return EpochLossStats(
        total_loss=running["total"] / samples,
        bpr_loss=running["bpr"] / samples,
        contrastive_loss=running["contrastive"] / samples,
        l2_loss=running["l2"] / samples,
    )


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    graphs: Any,
    train_ratings: np.ndarray,
    batch_size: int,
    device: torch.device,
    seed: int,
    epoch_number: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> EpochLossStats:
    """Train MNCL for one epoch."""
    model.train()
    loader = build_bpr_dataloader(
        ratings=train_ratings,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    running = _empty_running()
    progress = tqdm(loader, desc=f"Epoch {epoch_number} train", leave=False)

    for batch in progress:
        user_ids, positive_item_ids, negative_item_ids = _batch_to_device(
            batch,
            device,
        )

        optimizer.zero_grad(set_to_none=True)
        output = model(graphs)

        positive_scores = _pair_scores(
            output.user_embeddings,
            output.item_embeddings,
            user_ids,
            positive_item_ids,
        )
        negative_scores = _pair_scores(
            output.user_embeddings,
            output.item_embeddings,
            user_ids,
            negative_item_ids,
        )
        contrastive_user_ids, contrastive_item_ids = _contrastive_ids(
            user_ids,
            positive_item_ids,
            negative_item_ids,
        )

        loss_output = loss_fn(
            model=model,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            views=output.views,
            contrastive_user_ids=contrastive_user_ids,
            contrastive_item_ids=contrastive_item_ids,
        )
        loss_output.total_loss.backward()
        optimizer.step()

        current_batch_size = int(user_ids.numel())
        _update_running(running, loss_output, current_batch_size)
        progress.set_postfix(loss=f"{float(loss_output.total_loss.detach()):.4f}")

    return _finalize_running(running)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    graphs: Any,
    test_ratings: np.ndarray,
    batch_size: int,
    device: torch.device,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> EpochLossStats:
    """Evaluate deterministic MNCL loss on test triplets."""
    model.eval()
    loader = build_bpr_dataloader(
        ratings=test_ratings,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    output = model(
        graphs,
        add_noise=False,
        structural_dropout=False,
        message_dropout=False,
    )
    running = _empty_running()

    for batch in loader:
        user_ids, positive_item_ids, negative_item_ids = _batch_to_device(
            batch,
            device,
        )
        positive_scores = _pair_scores(
            output.user_embeddings,
            output.item_embeddings,
            user_ids,
            positive_item_ids,
        )
        negative_scores = _pair_scores(
            output.user_embeddings,
            output.item_embeddings,
            user_ids,
            negative_item_ids,
        )
        contrastive_user_ids, contrastive_item_ids = _contrastive_ids(
            user_ids,
            positive_item_ids,
            negative_item_ids,
        )

        loss_output = loss_fn(
            model=model,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            views=output.views,
            contrastive_user_ids=contrastive_user_ids,
            contrastive_item_ids=contrastive_item_ids,
        )

        current_batch_size = int(user_ids.numel())
        _update_running(running, loss_output, current_batch_size)

    return _finalize_running(running)


def train_epochs(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    graphs: Any,
    train_ratings: np.ndarray,
    test_ratings: np.ndarray,
    num_epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int = 42,
    start_epoch: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    best_checkpoint_path: str | Path | None = None,
    latest_checkpoint_path: str | Path | None = None,
    total_train_loss: list[float] | None = None,
    total_test_loss: list[float] | None = None,
    best_test_loss: float | None = None,
    config: Any | None = None,
) -> tuple[list[float], list[float]]:
    """Train epochs, save checkpoints, and return new loss arrays."""
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")
    if start_epoch < 0:
        raise ValueError("start_epoch must be non-negative.")

    base_train_history = list(total_train_loss or [])
    base_test_history = list(total_test_loss or [])
    if len(base_train_history) != len(base_test_history):
        raise ValueError("Training and test histories must have equal length.")
    if base_train_history and len(base_train_history) != start_epoch:
        raise ValueError("start_epoch must match the supplied history length.")

    current_best = (
        float(best_test_loss)
        if best_test_loss is not None
        else min(base_test_history, default=float("inf"))
    )
    train_loss: list[float] = []
    test_loss: list[float] = []

    epoch_progress = tqdm(range(num_epochs), desc="MNCL training")
    for local_epoch in epoch_progress:
        epoch_number = start_epoch + local_epoch + 1

        train_stats = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            graphs=graphs,
            train_ratings=train_ratings,
            batch_size=batch_size,
            device=device,
            seed=seed + epoch_number,
            epoch_number=epoch_number,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_stats = evaluate_one_epoch(
            model=model,
            loss_fn=loss_fn,
            graphs=graphs,
            test_ratings=test_ratings,
            batch_size=batch_size,
            device=device,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        train_loss.append(train_stats.total_loss)
        test_loss.append(test_stats.total_loss)

        full_train_history = base_train_history + train_loss
        full_test_history = base_test_history + test_loss
        improved = test_stats.total_loss < current_best
        if improved:
            current_best = test_stats.total_loss
            if best_checkpoint_path is not None:
                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch_number,
                    best_test_loss=current_best,
                    train_history=full_train_history,
                    test_history=full_test_history,
                    config=config,
                )

        if latest_checkpoint_path is not None:
            save_checkpoint(
                path=latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch_number,
                best_test_loss=current_best,
                train_history=full_train_history,
                test_history=full_test_history,
                config=config,
            )

        epoch_progress.set_postfix(
            train=f"{train_stats.total_loss:.4f}",
            test=f"{test_stats.total_loss:.4f}",
            best=f"{current_best:.4f}",
        )

    return train_loss, test_loss
