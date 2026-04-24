from __future__ import annotations

from dataclasses import asdict, dataclass


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class PatchConfig:
    patch_size: int = 64
    stride_h: int = 5
    stride_w: int = 5

    @property
    def patch_shape(self) -> tuple[int, int]:
        return (self.patch_size, self.patch_size)

    @property
    def stride(self) -> tuple[int, int]:
        return (self.stride_h, self.stride_w)

    def validate(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be greater than zero.")
        if self.stride_h <= 0 or self.stride_w <= 0:
            raise ValueError("stride_h and stride_w must be greater than zero.")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "resunet"
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 48
    depth: int = 4
    growth_factor: int = 2
    dropout: float = 0.2
    batchnorm: bool = True

    def validate(self) -> None:
        if self.architecture != "resunet":
            raise ValueError("architecture must be 'resunet'.")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than zero.")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be greater than zero.")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be greater than zero.")
        if self.depth <= 0:
            raise ValueError("depth must be greater than zero.")
        if self.growth_factor <= 1:
            raise ValueError("growth_factor must be greater than one.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 150
    batch_size: int = 64
    num_patches: int = 100_000
    val_patch_fraction: float = 0.1
    vessel_patch_ratio: float = 0.6
    hard_negative_patch_ratio: float = 0.2
    min_fov_coverage: float = 0.6
    hard_negative_band_width: int = 3
    patience: int = 20
    seed: int = 42

    @property
    def random_fov_patch_ratio(self) -> float:
        return 1.0 - self.vessel_patch_ratio - self.hard_negative_patch_ratio

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.num_patches <= 0:
            raise ValueError("num_patches must be greater than zero.")
        if self.val_patch_fraction < 0.0:
            raise ValueError("val_patch_fraction must be non-negative.")
        if not 0.0 <= self.vessel_patch_ratio <= 1.0:
            raise ValueError("vessel_patch_ratio must be in the range [0, 1].")
        if not 0.0 <= self.hard_negative_patch_ratio <= 1.0:
            raise ValueError("hard_negative_patch_ratio must be in the range [0, 1].")
        if self.vessel_patch_ratio + self.hard_negative_patch_ratio > 1.0:
            raise ValueError(
                "vessel_patch_ratio + hard_negative_patch_ratio must be less than or equal to 1."
            )
        if not 0.0 <= self.min_fov_coverage <= 1.0:
            raise ValueError("min_fov_coverage must be in the range [0, 1].")
        if self.hard_negative_band_width < 0:
            raise ValueError("hard_negative_band_width must be non-negative.")
        if self.patience <= 0:
            raise ValueError("patience must be greater than zero.")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class LossConfig:
    loss: str = "weighted_bce_dice"
    pos_weight: float = 3.0
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    focal_tversky_gamma: float = 1.33
    loss_eps: float = 1e-6

    def validate(self) -> None:
        if self.loss not in {"bce_dice", "weighted_bce_dice", "tversky", "focal_tversky"}:
            raise ValueError(
                "loss must be one of: bce_dice, weighted_bce_dice, tversky, focal_tversky."
            )
        if self.loss_eps <= 0.0:
            raise ValueError("loss_eps must be greater than zero.")
        if self.loss in {"bce_dice", "weighted_bce_dice"}:
            if self.dice_weight < 0.0:
                raise ValueError("dice_weight must be non-negative.")
            if self.bce_weight < 0.0:
                raise ValueError("bce_weight must be non-negative.")
            if self.bce_weight + self.dice_weight <= 0.0:
                raise ValueError("bce_weight + dice_weight must be greater than zero.")
            if self.loss == "weighted_bce_dice" and self.pos_weight <= 0.0:
                raise ValueError("pos_weight must be greater than zero.")
        if self.loss in {"tversky", "focal_tversky"}:
            if self.tversky_alpha < 0.0:
                raise ValueError("tversky_alpha must be non-negative.")
            if self.tversky_beta < 0.0:
                raise ValueError("tversky_beta must be non-negative.")
            if self.tversky_alpha + self.tversky_beta <= 0.0:
                raise ValueError("tversky_alpha + tversky_beta must be greater than zero.")
            if self.loss == "focal_tversky" and self.focal_tversky_gamma <= 0.0:
                raise ValueError("focal_tversky_gamma must be greater than zero.")

    def to_dict(self) -> dict[str, float | str]:
        summary: dict[str, float | str] = {
            "loss": self.loss,
            "loss_eps": self.loss_eps,
        }
        if self.loss in {"bce_dice", "weighted_bce_dice"}:
            summary["bce_weight"] = self.bce_weight
            summary["dice_weight"] = self.dice_weight
            if self.loss == "weighted_bce_dice":
                summary["pos_weight"] = self.pos_weight
        if self.loss in {"tversky", "focal_tversky"}:
            summary["tversky_alpha"] = self.tversky_alpha
            summary["tversky_beta"] = self.tversky_beta
            if self.loss == "focal_tversky":
                summary["focal_tversky_gamma"] = self.focal_tversky_gamma
        return summary


@dataclass(frozen=True)
class ValidationConfig:
    selection_metric: str = "val_dice"
    val_threshold: float = 0.5
    run_full_image_validation: bool = True
    full_image_validation_every: int = 1

    @property
    def selection_metric_source(self) -> str:
        if self.selection_metric == "val_loss":
            return "patch_validation_loss"
        return "full_image_validation"

    def validate(self) -> None:
        if self.selection_metric not in {"val_loss", "val_dice", "val_f1"}:
            raise ValueError("selection_metric must be one of: val_loss, val_dice, val_f1.")
        if not 0.0 <= self.val_threshold <= 1.0:
            raise ValueError("val_threshold must be in the range [0, 1].")
        if self.full_image_validation_every <= 0:
            raise ValueError("full_image_validation_every must be greater than zero.")
        if not self.run_full_image_validation and self.selection_metric != "val_loss":
            raise ValueError(
                "Full-image validation must be enabled when selection_metric is val_dice or val_f1."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_metric": self.selection_metric,
            "selection_metric_source": self.selection_metric_source,
            "val_threshold": self.val_threshold,
            "run_full_image_validation": self.run_full_image_validation,
            "full_image_validation_every": self.full_image_validation_every,
        }


@dataclass(frozen=True)
class OptimizationConfig:
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    def validate(self) -> None:
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be one of: adamw, sgd.")
        if self.lr <= 0.0:
            raise ValueError("lr must be greater than zero.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.momentum < 0.0:
            raise ValueError("momentum must be non-negative.")
        if self.scheduler not in {"none", "reduce_on_plateau", "cosine"}:
            raise ValueError("scheduler must be one of: none, reduce_on_plateau, cosine.")
        if self.scheduler_patience < 0:
            raise ValueError("scheduler_patience must be non-negative.")
        if self.min_lr < 0.0:
            raise ValueError("min_lr must be non-negative.")
        if self.min_lr > self.lr:
            raise ValueError("min_lr must be less than or equal to lr.")
        if self.scheduler == "reduce_on_plateau" and not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be in the range (0, 1).")

    def to_dict(self, total_epochs: int | None = None) -> dict[str, object]:
        summary: dict[str, object] = {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "scheduler": self.scheduler,
        }
        if self.optimizer == "sgd":
            summary["momentum"] = self.momentum

        if self.scheduler == "none":
            summary["scheduler_settings"] = {}
        elif self.scheduler == "reduce_on_plateau":
            summary["scheduler_settings"] = {
                "monitor": "val_loss",
                "patience": self.scheduler_patience,
                "factor": self.scheduler_factor,
                "min_lr": self.min_lr,
            }
        else:
            scheduler_settings: dict[str, object] = {
                "min_lr": self.min_lr,
            }
            if total_epochs is not None:
                scheduler_settings["epochs"] = total_epochs
            summary["scheduler_settings"] = scheduler_settings

        return summary


@dataclass(frozen=True)
class AugmentationConfig:
    enable_augmentation: bool = True
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rot90_prob: float = 0.5
    max_rotation_deg: float = 15.0
    rotation_prob: float = 0.3
    brightness_jitter: float = 0.1
    contrast_jitter: float = 0.1
    gamma_jitter: float = 0.1
    noise_prob: float = 0.2
    noise_std: float = 0.02
    blur_prob: float = 0.1
    blur_kernel_size: int = 3

    def validate(self) -> None:
        for name, value in (
            ("hflip_prob", self.hflip_prob),
            ("vflip_prob", self.vflip_prob),
            ("rot90_prob", self.rot90_prob),
            ("rotation_prob", self.rotation_prob),
            ("noise_prob", self.noise_prob),
            ("blur_prob", self.blur_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in the range [0, 1].")

        for name, value in (
            ("max_rotation_deg", self.max_rotation_deg),
            ("brightness_jitter", self.brightness_jitter),
            ("contrast_jitter", self.contrast_jitter),
            ("gamma_jitter", self.gamma_jitter),
            ("noise_std", self.noise_std),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        if self.blur_kernel_size <= 0 or self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be a positive odd integer.")

    def to_dict(self) -> dict[str, bool | int | float]:
        return asdict(self)
