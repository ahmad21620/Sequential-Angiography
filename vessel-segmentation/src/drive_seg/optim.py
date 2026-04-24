from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from .config import OptimizationConfig


SchedulerType = (
    torch.optim.lr_scheduler.CosineAnnealingLR
    | torch.optim.lr_scheduler.ReduceLROnPlateau
)


@dataclass
class OptimizationStack:
    optimizer: torch.optim.Optimizer
    scheduler: SchedulerType | None
    scheduler_name: str

    def step_scheduler(self, val_loss: float) -> None:
        if self.scheduler is None:
            return
        if self.scheduler_name == "reduce_on_plateau":
            self.scheduler.step(val_loss)
            return
        self.scheduler.step()

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def scheduler_state_dict(self) -> dict[str, Any] | None:
        if self.scheduler is None:
            return None
        return self.scheduler.state_dict()


def build_optimization_stack(
    parameters: Iterable[torch.nn.Parameter],
    config: OptimizationConfig,
    epochs: int,
) -> OptimizationStack:
    optimizer = _build_optimizer(parameters, config)
    scheduler = _build_scheduler(optimizer, config, epochs)
    return OptimizationStack(
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_name=config.scheduler,
    )


def _build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: OptimizationConfig,
) -> torch.optim.Optimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: OptimizationConfig,
    epochs: int,
) -> SchedulerType | None:
    if config.scheduler == "none":
        return None
    if config.scheduler == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_lr,
        )
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=config.min_lr,
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")
