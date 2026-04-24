from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig


def _flatten_binary_tensors(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = probabilities.float().flatten(start_dim=1)
    targets = targets.float().flatten(start_dim=1)
    return probabilities, targets


def _dice_loss_from_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    probabilities, targets = _flatten_binary_tensors(probabilities, targets)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice_score.mean()


def _tversky_score_from_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float,
) -> torch.Tensor:
    probabilities, targets = _flatten_binary_tensors(probabilities, targets)
    true_positive = (probabilities * targets).sum(dim=1)
    false_positive = (probabilities * (1.0 - targets)).sum(dim=1)
    false_negative = ((1.0 - probabilities) * targets).sum(dim=1)
    return (true_positive + eps) / (
        true_positive + alpha * false_positive + beta * false_negative + eps
    )


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        return _dice_loss_from_probabilities(probabilities, targets, eps=self.eps)


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        eps: float = 1e-6,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(eps=eps)
        if pos_weight is None:
            self.use_pos_weight = False
            self.register_buffer("_pos_weight", torch.tensor(1.0, dtype=torch.float32))
        else:
            self.use_pos_weight = True
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self._pos_weight if self.use_pos_weight else None,
        )
        dice = self.dice(logits, targets)
        return (self.bce_weight * bce) + (self.dice_weight * dice)


class TverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        tversky_score = _tversky_score_from_probabilities(
            probabilities,
            targets,
            alpha=self.alpha,
            beta=self.beta,
            eps=self.eps,
        )
        return 1.0 - tversky_score.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        tversky_score = _tversky_score_from_probabilities(
            probabilities,
            targets,
            alpha=self.alpha,
            beta=self.beta,
            eps=self.eps,
        )
        focal_term = torch.clamp(1.0 - tversky_score, min=0.0)
        return torch.pow(focal_term, self.gamma).mean()


def build_loss(config: LossConfig) -> nn.Module:
    if config.loss == "bce_dice":
        return BCEDiceLoss(
            bce_weight=config.bce_weight,
            dice_weight=config.dice_weight,
            eps=config.loss_eps,
            pos_weight=None,
        )
    if config.loss == "weighted_bce_dice":
        return BCEDiceLoss(
            bce_weight=config.bce_weight,
            dice_weight=config.dice_weight,
            eps=config.loss_eps,
            pos_weight=config.pos_weight,
        )
    if config.loss == "tversky":
        return TverskyLoss(
            alpha=config.tversky_alpha,
            beta=config.tversky_beta,
            eps=config.loss_eps,
        )
    if config.loss == "focal_tversky":
        return FocalTverskyLoss(
            alpha=config.tversky_alpha,
            beta=config.tversky_beta,
            gamma=config.focal_tversky_gamma,
            eps=config.loss_eps,
        )
    raise ValueError(f"Unsupported loss: {config.loss}")
