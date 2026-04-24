from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drive_seg.augmentations import PatchAugmenter
from drive_seg.config import (
    AugmentationConfig,
    LossConfig,
    ModelConfig,
    OptimizationConfig,
    PatchConfig,
    TrainingConfig,
    ValidationConfig,
)
from drive_seg.dataset import (
    DatasetValidationError,
    LazyRandomPatchDataset,
    RandomPatchDataset,
    SegmentationSamplePaths,
    SegmentationSplit,
    format_validation_report,
    load_dataset_split,
    load_sample_fov_mask,
    load_sample_image,
    load_sample_mask,
    validate_dataset_root,
)
from drive_seg.engine import FullImageValidationData, TrainingResumeState, fit_model
from drive_seg.losses import build_loss
from drive_seg.model import build_model
from drive_seg.optim import build_optimization_stack
from drive_seg.patching import sample_patch_coordinates, sample_patch_coordinates_streaming
from drive_seg.preprocessing import binarize_masks, preprocess_image
from drive_seg.utils import ensure_dir, load_checkpoint, resolve_device, save_json, set_seed
from drive_seg.visualization import save_training_history


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def build_parser() -> argparse.ArgumentParser:
    patch_defaults = PatchConfig()
    model_defaults = ModelConfig()
    training_defaults = TrainingConfig()
    validation_defaults = ValidationConfig()
    loss_defaults = LossConfig()
    optimization_defaults = OptimizationConfig()
    augmentation_defaults = AugmentationConfig()

    parser = argparse.ArgumentParser(
        description="Train a PyTorch ResUNet on explicit vessel-segmentation dataset splits."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing split folders such as train/, val/, and test/.",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Name of the dataset split used for training.",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="val",
        help="Name of the dataset split used for validation.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("val_loss", "val_dice", "val_f1"),
        default=validation_defaults.selection_metric,
        help="Metric used for best-checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--val-threshold",
        type=float,
        default=validation_defaults.val_threshold,
        help="Threshold applied to full-image validation probabilities before metric computation.",
    )
    parser.add_argument(
        "--run-full-image-validation",
        type=parse_bool,
        default=validation_defaults.run_full_image_validation,
        help="Run deterministic full-image validation on the validation split (true/false).",
    )
    parser.add_argument(
        "--full-image-validation-every",
        type=int,
        default=validation_defaults.full_image_validation_every,
        help="Run full-image validation every N epochs while keeping patch validation every epoch.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "best_model.pt",
        help="Where to save the best model checkpoint.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint used to initialize model weights before training. This loads weights only, not optimizer/scheduler state.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint used to resume training, including model, optimizer, scheduler, history, and epoch state.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=training_defaults.epochs,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=training_defaults.batch_size,
        help="Batch size for patch training.",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        default=training_defaults.num_patches,
        help="Total stratified training patches sampled from the training split.",
    )
    parser.add_argument(
        "--val-patch-fraction",
        type=float,
        default=training_defaults.val_patch_fraction,
        help="Controls the number of validation patches sampled from the validation split relative to --num-patches.",
    )
    parser.add_argument(
        "--vessel-patch-ratio",
        type=float,
        default=training_defaults.vessel_patch_ratio,
        help="Target fraction of vessel-centered patches in the stratified sampler.",
    )
    parser.add_argument(
        "--hard-negative-patch-ratio",
        type=float,
        default=training_defaults.hard_negative_patch_ratio,
        help="Target fraction of near-vessel background patches in the stratified sampler.",
    )
    parser.add_argument(
        "--min-fov-coverage",
        type=float,
        default=training_defaults.min_fov_coverage,
        help="Minimum fraction of valid-region pixels required inside each sampled patch. Uses provided FOV masks when available and full-image masks otherwise.",
    )
    parser.add_argument(
        "--hard-negative-band-width",
        type=int,
        default=training_defaults.hard_negative_band_width,
        help="Width of the near-vessel band used to define hard-negative centers.",
    )
    parser.add_argument(
        "--enable-augmentation",
        type=parse_bool,
        default=augmentation_defaults.enable_augmentation,
        help="Enable conservative training-time patch augmentation (true/false).",
    )
    parser.add_argument(
        "--hflip-prob",
        type=float,
        default=augmentation_defaults.hflip_prob,
        help="Probability of a horizontal flip during training augmentation.",
    )
    parser.add_argument(
        "--vflip-prob",
        type=float,
        default=augmentation_defaults.vflip_prob,
        help="Probability of a vertical flip during training augmentation.",
    )
    parser.add_argument(
        "--rot90-prob",
        type=float,
        default=augmentation_defaults.rot90_prob,
        help="Probability of applying a random 90-degree multiple rotation during training augmentation.",
    )
    parser.add_argument(
        "--max-rotation-deg",
        type=float,
        default=augmentation_defaults.max_rotation_deg,
        help="Maximum absolute angle for mild free-angle rotations during training augmentation.",
    )
    parser.add_argument(
        "--rotation-prob",
        type=float,
        default=augmentation_defaults.rotation_prob,
        help="Probability of applying a mild free-angle rotation during training augmentation.",
    )
    parser.add_argument(
        "--brightness-jitter",
        type=float,
        default=augmentation_defaults.brightness_jitter,
        help="Maximum brightness adjustment applied to training patches.",
    )
    parser.add_argument(
        "--contrast-jitter",
        type=float,
        default=augmentation_defaults.contrast_jitter,
        help="Maximum contrast adjustment applied to training patches.",
    )
    parser.add_argument(
        "--gamma-jitter",
        type=float,
        default=augmentation_defaults.gamma_jitter,
        help="Maximum gamma adjustment applied to training patches.",
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=augmentation_defaults.noise_prob,
        help="Probability of adding mild Gaussian noise to a training patch.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=augmentation_defaults.noise_std,
        help="Standard deviation of Gaussian noise added to training patches.",
    )
    parser.add_argument(
        "--blur-prob",
        type=float,
        default=augmentation_defaults.blur_prob,
        help="Probability of applying mild blur to a training patch.",
    )
    parser.add_argument(
        "--blur-kernel-size",
        type=int,
        default=augmentation_defaults.blur_kernel_size,
        help="Kernel size for mild Gaussian blur applied to training patches.",
    )
    parser.add_argument(
        "--loss",
        choices=("bce_dice", "weighted_bce_dice", "tversky", "focal_tversky"),
        default=loss_defaults.loss,
        help="Loss used for binary vessel segmentation training.",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=loss_defaults.pos_weight,
        help="Positive-class weight used by the weighted BCE + Dice loss.",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=loss_defaults.dice_weight,
        help="Dice weight used by the BCE + Dice loss variants.",
    )
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=loss_defaults.bce_weight,
        help="BCE weight used by the BCE + Dice loss variants.",
    )
    parser.add_argument(
        "--tversky-alpha",
        type=float,
        default=loss_defaults.tversky_alpha,
        help="Alpha parameter used by the Tversky-based losses.",
    )
    parser.add_argument(
        "--tversky-beta",
        type=float,
        default=loss_defaults.tversky_beta,
        help="Beta parameter used by the Tversky-based losses.",
    )
    parser.add_argument(
        "--focal-tversky-gamma",
        type=float,
        default=loss_defaults.focal_tversky_gamma,
        help="Gamma parameter used by the focal Tversky loss.",
    )
    parser.add_argument(
        "--loss-eps",
        type=float,
        default=loss_defaults.loss_eps,
        help="Numerical stability epsilon used by Dice and Tversky-style losses.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "sgd"),
        default=optimization_defaults.optimizer,
        help="Optimizer used for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=optimization_defaults.lr,
        help="Base learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=optimization_defaults.weight_decay,
        help="Weight decay applied by the optimizer.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=optimization_defaults.momentum,
        help="Momentum used when --optimizer sgd.",
    )
    parser.add_argument(
        "--scheduler",
        choices=("none", "reduce_on_plateau", "cosine"),
        default=optimization_defaults.scheduler,
        help="Learning-rate scheduler used during training.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=optimization_defaults.scheduler_patience,
        help="Patience for the reduce_on_plateau scheduler.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=optimization_defaults.scheduler_factor,
        help="Multiplicative decay factor for the reduce_on_plateau scheduler.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=optimization_defaults.min_lr,
        help="Minimum learning rate allowed by the scheduler.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=training_defaults.patience,
        help="Early stopping patience based on the configured selection metric.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=training_defaults.seed,
        help="Random seed used for patch sampling, augmentation, and training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers used for patch loading.",
    )
    parser.add_argument(
        "--lazy-train-loading",
        type=parse_bool,
        default=True,
        help="Load training images lazily from disk instead of preloading the full training split into RAM (true/false).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=patch_defaults.patch_size,
        help="Square patch size used for training, validation, and checkpointed inference geometry.",
    )
    parser.add_argument(
        "--stride-h",
        type=int,
        default=patch_defaults.stride_h,
        help="Stored in the checkpoint for later evaluation and inference.",
    )
    parser.add_argument(
        "--stride-w",
        type=int,
        default=patch_defaults.stride_w,
        help="Stored in the checkpoint for later evaluation and inference.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=model_defaults.base_channels,
        help="ResUNet channel width at the first encoder stage. Default 48; try 64 if memory allows.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=model_defaults.depth,
        help="Number of downsampling stages in the ResUNet.",
    )
    parser.add_argument(
        "--growth-factor",
        type=int,
        default=model_defaults.growth_factor,
        help="Channel multiplier between encoder stages.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=model_defaults.dropout,
        help="Dropout probability used inside residual blocks.",
    )
    parser.add_argument(
        "--no-batchnorm",
        action="store_true",
        help="Disable batch normalization inside the ResUNet blocks.",
    )
    return parser


def load_preprocessed_split(
    split: SegmentationSplit,
    split_label: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    print(f"Loading {split_label} split into memory...")
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    fov_masks: list[np.ndarray] = []
    for sample in split.samples:
        image = preprocess_image(load_sample_image(sample))
        mask = binarize_masks(load_sample_mask(sample))
        fov_mask = binarize_masks(load_sample_fov_mask(sample, spatial_shape=mask.shape[-2:]))
        images.append(image)
        masks.append(mask)
        fov_masks.append(fov_mask)
    return images, masks, fov_masks


def load_sampling_mask_pair(
    sample: SegmentationSamplePaths,
) -> tuple[np.ndarray, np.ndarray]:
    mask = binarize_masks(load_sample_mask(sample))
    fov_mask = binarize_masks(load_sample_fov_mask(sample, spatial_shape=mask.shape[-2:]))
    return mask, fov_mask


def group_coordinates_by_image(
    coordinates: np.ndarray,
    seed: int,
) -> np.ndarray:
    if coordinates.shape[0] <= 1:
        return coordinates.astype(np.int32, copy=False)

    unique_images = np.unique(coordinates[:, 0]).astype(np.int32, copy=False)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_images)

    max_image_index = int(unique_images.max())
    image_order = np.full(max_image_index + 1, unique_images.size, dtype=np.int32)
    image_order[unique_images] = np.arange(unique_images.size, dtype=np.int32)
    sort_order = np.lexsort(
        (
            coordinates[:, 2],
            coordinates[:, 1],
            image_order[coordinates[:, 0]],
        )
    )
    return coordinates[sort_order].astype(np.int32, copy=False)


def ensure_disjoint_image_ids(
    train_ids: list[str],
    val_ids: list[str],
    train_split: str,
    val_split: str,
) -> None:
    overlap = sorted(set(train_ids) & set(val_ids))
    if not overlap:
        return

    overlap_preview = ", ".join(overlap[:5])
    if len(overlap) > 5:
        overlap_preview = f"{overlap_preview}, ..."
    raise ValueError(
        "Training and validation image IDs must be disjoint. "
        f"Overlap between split '{train_split}' and split '{val_split}': {overlap_preview}"
    )


def initialize_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} does not contain model_state_dict and cannot be used with --init-checkpoint."
        )
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            f"{checkpoint_path} is not compatible with the current model configuration. "
            "Use matching architecture settings or a compatible initialization checkpoint."
        ) from exc
    return checkpoint


def _derive_epochs_without_improvement(checkpoint: dict[str, object]) -> int:
    if "epochs_without_improvement" in checkpoint:
        return int(checkpoint["epochs_without_improvement"])

    completed_epoch = int(checkpoint.get("epoch", 0))
    history = checkpoint.get("history")
    if isinstance(history, dict):
        best_epoch = history.get("best_epoch")
        if best_epoch is None:
            return 0
        return max(0, completed_epoch - int(best_epoch))
    return 0


def resume_training_from_checkpoint(
    model: torch.nn.Module,
    optimization,
    checkpoint_path: Path,
    device: torch.device,
    validation_config: ValidationConfig,
) -> TrainingResumeState:
    checkpoint = load_checkpoint(checkpoint_path, device)
    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "history",
        "epoch",
        "best_selection_metric",
        "best_selection_value",
    }
    missing_keys = sorted(required_keys - set(checkpoint))
    if missing_keys:
        missing_list = ", ".join(missing_keys)
        raise ValueError(
            f"{checkpoint_path} cannot be used with --resume-checkpoint because it is missing: {missing_list}"
        )

    checkpoint_selection_metric = str(checkpoint["best_selection_metric"])
    if checkpoint_selection_metric != validation_config.selection_metric:
        raise ValueError(
            f"{checkpoint_path} was trained with selection metric '{checkpoint_selection_metric}', "
            f"but the current run requested '{validation_config.selection_metric}'. "
            "Use the same selection metric when resuming."
        )

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            f"{checkpoint_path} is not compatible with the current model configuration."
        ) from exc

    try:
        optimization.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except ValueError as exc:
        raise ValueError(
            f"{checkpoint_path} is not compatible with the current optimizer configuration."
        ) from exc

    if optimization.scheduler is not None:
        scheduler_state_dict = checkpoint.get("scheduler_state_dict")
        if scheduler_state_dict is None:
            raise ValueError(
                f"{checkpoint_path} does not contain scheduler state, but the current run expects scheduler '{optimization.scheduler_name}'."
            )
        optimization.scheduler.load_state_dict(scheduler_state_dict)
    elif "scheduler_state_dict" in checkpoint:
        raise ValueError(
            f"{checkpoint_path} contains scheduler state, but the current run requested scheduler '{optimization.scheduler_name}'."
        )

    history = checkpoint["history"]
    if not isinstance(history, dict):
        raise ValueError(f"{checkpoint_path} has an invalid training history payload.")

    start_epoch = int(checkpoint["epoch"]) + 1
    return TrainingResumeState(
        start_epoch=start_epoch,
        history=history,
        best_selection_value=float(checkpoint["best_selection_value"]),
        epochs_without_improvement=_derive_epochs_without_improvement(checkpoint),
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.train_split == args.val_split:
        raise ValueError("--train-split and --val-split must be different explicit splits.")
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError("--init-checkpoint and --resume-checkpoint cannot be used together.")

    set_seed(args.seed)
    device = resolve_device(args.device)

    patch_config = PatchConfig(
        patch_size=args.patch_size,
        stride_h=args.stride_h,
        stride_w=args.stride_w,
    )
    model_config = ModelConfig(
        base_channels=args.base_channels,
        depth=args.depth,
        growth_factor=args.growth_factor,
        dropout=args.dropout,
        batchnorm=not args.no_batchnorm,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_patches=args.num_patches,
        val_patch_fraction=args.val_patch_fraction,
        vessel_patch_ratio=args.vessel_patch_ratio,
        hard_negative_patch_ratio=args.hard_negative_patch_ratio,
        min_fov_coverage=args.min_fov_coverage,
        hard_negative_band_width=args.hard_negative_band_width,
        patience=args.patience,
        seed=args.seed,
    )
    validation_config = ValidationConfig(
        selection_metric=args.selection_metric,
        val_threshold=args.val_threshold,
        run_full_image_validation=args.run_full_image_validation,
        full_image_validation_every=args.full_image_validation_every,
    )
    loss_config = LossConfig(
        loss=args.loss,
        pos_weight=args.pos_weight,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        focal_tversky_gamma=args.focal_tversky_gamma,
        loss_eps=args.loss_eps,
    )
    optimization_config = OptimizationConfig(
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        scheduler=args.scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
    )
    augmentation_config = AugmentationConfig(
        enable_augmentation=args.enable_augmentation,
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        rot90_prob=args.rot90_prob,
        max_rotation_deg=args.max_rotation_deg,
        rotation_prob=args.rotation_prob,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
        gamma_jitter=args.gamma_jitter,
        noise_prob=args.noise_prob,
        noise_std=args.noise_std,
        blur_prob=args.blur_prob,
        blur_kernel_size=args.blur_kernel_size,
    )
    patch_config.validate()
    model_config.validate()
    training_config.validate()
    validation_config.validate()
    loss_config.validate()
    optimization_config.validate()
    augmentation_config.validate()

    dataset_root = args.dataset_root.resolve()
    requested_splits = tuple(dict.fromkeys((args.train_split, args.val_split)))
    dataset_report = validate_dataset_root(dataset_root, splits=requested_splits)
    print(format_validation_report(dataset_report))
    if not dataset_report.is_valid:
        raise DatasetValidationError("Dataset validation failed. Fix the reported issues and retry.")

    train_dataset_split = load_dataset_split(dataset_root, args.train_split)
    val_dataset_split = load_dataset_split(dataset_root, args.val_split)
    ensure_disjoint_image_ids(
        train_dataset_split.image_ids,
        val_dataset_split.image_ids,
        args.train_split,
        args.val_split,
    )

    validation_patch_count = max(
        training_config.batch_size,
        int(round(training_config.num_patches * training_config.val_patch_fraction)),
    )

    split_report_by_name = {
        split_report.split_name: {
            "original_count": split_report.original_count,
            "mask_count": split_report.mask_count,
            "fov_count": split_report.fov_count,
            "paired_count": split_report.paired_count,
            "fov_mode": split_report.fov_mode,
        }
        for split_report in dataset_report.split_reports
    }
    dataset_summary: dict[str, object] = {
        "dataset_root": dataset_root,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "initialization": {
            "mode": (
                "resume"
                if args.resume_checkpoint is not None
                else "checkpoint"
                if args.init_checkpoint is not None
                else "scratch"
            ),
            "init_checkpoint": args.init_checkpoint.resolve() if args.init_checkpoint is not None else None,
            "resume_checkpoint": args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None,
        },
        "split_counts": split_report_by_name,
        "train_image_ids": train_dataset_split.image_ids,
        "validation_image_ids": val_dataset_split.image_ids,
        "train_image_count": len(train_dataset_split.image_ids),
        "validation_image_count": len(val_dataset_split.image_ids),
        "train_patch_count": training_config.num_patches,
        "validation_patch_count": validation_patch_count,
        "train_data_loading_mode": "lazy" if args.lazy_train_loading else "preloaded",
        "patch": {
            "patch_size": patch_config.patch_size,
            "stride_h": patch_config.stride_h,
            "stride_w": patch_config.stride_w,
        },
        "validation": validation_config.to_dict(),
        "patch_sampling": {
            "vessel_patch_ratio": training_config.vessel_patch_ratio,
            "hard_negative_patch_ratio": training_config.hard_negative_patch_ratio,
            "random_fov_patch_ratio": training_config.random_fov_patch_ratio,
            "min_fov_coverage": training_config.min_fov_coverage,
            "hard_negative_band_width": training_config.hard_negative_band_width,
        },
        "loss": loss_config.to_dict(),
        "optimization": optimization_config.to_dict(total_epochs=training_config.epochs),
        "augmentation": augmentation_config.to_dict(),
    }

    print(
        f"Training images available for stratified patch sampling: {len(train_dataset_split.samples)}"
    )
    print(
        f"Validation images available for stratified patch sampling: {len(val_dataset_split.samples)}"
    )

    if args.lazy_train_loading:
        print("Sampling training patch coordinates from split files...")
        train_sampling = sample_patch_coordinates_streaming(
            image_count=len(train_dataset_split.samples),
            load_mask_pair=lambda image_index: load_sampling_mask_pair(
                train_dataset_split.samples[image_index]
            ),
            patch_size=patch_config.patch_shape,
            num_patches=training_config.num_patches,
            seed=training_config.seed,
            vessel_patch_ratio=training_config.vessel_patch_ratio,
            hard_negative_patch_ratio=training_config.hard_negative_patch_ratio,
            min_fov_coverage=training_config.min_fov_coverage,
            hard_negative_band_width=training_config.hard_negative_band_width,
        )
        train_coordinates = group_coordinates_by_image(
            train_sampling.coordinates,
            seed=training_config.seed,
        )
        train_images = None
        train_masks = None
    else:
        train_images, train_masks, train_fov_masks = load_preprocessed_split(
            train_dataset_split,
            split_label=args.train_split,
        )
        print("Sampling training patch coordinates...")
        train_sampling = sample_patch_coordinates(
            masks=train_masks,
            fov_masks=train_fov_masks,
            patch_size=patch_config.patch_shape,
            num_patches=training_config.num_patches,
            seed=training_config.seed,
            vessel_patch_ratio=training_config.vessel_patch_ratio,
            hard_negative_patch_ratio=training_config.hard_negative_patch_ratio,
            min_fov_coverage=training_config.min_fov_coverage,
            hard_negative_band_width=training_config.hard_negative_band_width,
        )
        train_coordinates = train_sampling.coordinates

    validation_images, validation_masks, validation_fov_masks = load_preprocessed_split(
        val_dataset_split,
        split_label=args.val_split,
    )
    if not validation_images:
        raise ValueError("Validation split is empty.")

    print("Sampling validation patch coordinates...")
    validation_sampling = sample_patch_coordinates(
        masks=validation_masks,
        fov_masks=validation_fov_masks,
        patch_size=patch_config.patch_shape,
        num_patches=validation_patch_count,
        seed=training_config.seed + 1,
        vessel_patch_ratio=training_config.vessel_patch_ratio,
        hard_negative_patch_ratio=training_config.hard_negative_patch_ratio,
        min_fov_coverage=training_config.min_fov_coverage,
        hard_negative_band_width=training_config.hard_negative_band_width,
    )

    dataset_summary["train_patch_category_counts"] = train_sampling.category_counts
    dataset_summary["validation_patch_category_counts"] = validation_sampling.category_counts

    train_augmenter = (
        PatchAugmenter(augmentation_config) if augmentation_config.enable_augmentation else None
    )
    if args.lazy_train_loading:
        train_dataset = LazyRandomPatchDataset(
            samples=train_dataset_split.samples,
            coordinates=train_coordinates,
            patch_size=patch_config.patch_shape,
            augmenter=train_augmenter,
            max_cached_images=max(2, args.num_workers + 1),
        )
        train_loader_shuffle = False
    else:
        train_dataset = RandomPatchDataset(
            images=train_images,
            masks=train_masks,
            coordinates=train_coordinates,
            patch_size=patch_config.patch_shape,
            augmenter=train_augmenter,
        )
        train_loader_shuffle = True
    val_dataset = RandomPatchDataset(
        images=validation_images,
        masks=validation_masks,
        coordinates=validation_sampling.coordinates,
        patch_size=patch_config.patch_shape,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        # Keep grouped coordinates in order when lazy loading so workers can reuse cached source images.
        shuffle=train_loader_shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(model_config).to(device)
    init_checkpoint_metadata: dict[str, object] | None = None
    resume_state: TrainingResumeState | None = None
    if args.init_checkpoint is not None:
        init_checkpoint_metadata = initialize_model_from_checkpoint(
            model=model,
            checkpoint_path=args.init_checkpoint,
            device=device,
        )
        print(f"Initialized model weights from {args.init_checkpoint}")
    criterion = build_loss(loss_config).to(device)
    optimization = build_optimization_stack(
        model.parameters(),
        config=optimization_config,
        epochs=training_config.epochs,
    )
    if args.resume_checkpoint is not None:
        resume_state = resume_training_from_checkpoint(
            model=model,
            optimization=optimization,
            checkpoint_path=args.resume_checkpoint,
            device=device,
            validation_config=validation_config,
        )
        print(
            f"Resumed training state from {args.resume_checkpoint} "
            f"(next epoch: {resume_state.start_epoch})"
        )
    full_image_validation_data = FullImageValidationData(
        images=validation_images,
        masks=validation_masks,
        fov_masks=validation_fov_masks,
        patch_size=patch_config.patch_shape,
        stride=patch_config.stride,
        batch_size=training_config.batch_size,
        num_workers=args.num_workers,
    )

    checkpoint_path = args.checkpoint
    last_checkpoint_path = checkpoint_path.with_name("last_model.pt")
    run_config = training_config.to_dict()
    run_config["dataset"] = {
        "dataset_root": dataset_root,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "train_image_count": len(train_dataset_split.image_ids),
        "validation_image_count": len(val_dataset_split.image_ids),
        "train_fov_mode": train_dataset_split.fov_mode,
        "validation_fov_mode": val_dataset_split.fov_mode,
        "train_data_loading_mode": "lazy" if args.lazy_train_loading else "preloaded",
    }
    run_config["initialization"] = {
        "mode": (
            "resume"
            if args.resume_checkpoint is not None
            else "checkpoint"
            if args.init_checkpoint is not None
            else "scratch"
        ),
        "init_checkpoint": args.init_checkpoint.resolve() if args.init_checkpoint is not None else None,
        "resume_checkpoint": args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None,
        "init_model_config": init_checkpoint_metadata.get("model_config") if init_checkpoint_metadata is not None else None,
    }
    run_config["validation"] = validation_config.to_dict()
    run_config["loss"] = loss_config.to_dict()
    run_config["optimization"] = optimization_config.to_dict(total_epochs=training_config.epochs)
    run_config["augmentation"] = augmentation_config.to_dict()
    history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimization=optimization,
        criterion=criterion,
        device=device,
        epochs=training_config.epochs,
        patience=training_config.patience,
        validation_config=validation_config,
        full_image_validation_data=full_image_validation_data if validation_config.run_full_image_validation else None,
        checkpoint_path=checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        model_config=model_config.to_dict(),
        patch_config=patch_config.to_dict(),
        training_config=run_config,
        resume_state=resume_state,
    )

    history_dir = ensure_dir(ROOT / "outputs" / "training")
    save_training_history(history, history_dir / "learning_curve.png")
    save_json(history_dir / "history.json", history)
    save_json(history_dir / "dataset_summary.json", dataset_summary)

    print(f"Best checkpoint saved to {checkpoint_path}")
    print(f"Last checkpoint saved to {last_checkpoint_path}")
    print(f"Training curves saved to {history_dir / 'learning_curve.png'}")
    print(f"Dataset summary saved to {history_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
