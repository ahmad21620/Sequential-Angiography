from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import ValidationConfig
from .dataset import PatchArrayDataset
from .metrics import compute_binary_segmentation_metrics, dice_coefficient_from_logits
from .optim import OptimizationStack
from .patching import (
    extract_ordered_patches,
    pad_image_for_patching,
    reconstruct_from_ordered_patches,
)
from .utils import ensure_dir


def _autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


@dataclass(frozen=True)
class FullImageValidationData:
    images: Sequence[np.ndarray]
    masks: Sequence[np.ndarray]
    fov_masks: Sequence[np.ndarray] | None
    patch_size: tuple[int, int]
    stride: tuple[int, int]
    batch_size: int
    num_workers: int = 0

    def validate(self) -> None:
        image_count = len(self.images)
        if image_count == 0:
            raise ValueError("Full-image validation requires at least one validation image.")
        if len(self.masks) != image_count:
            raise ValueError("Full-image validation masks must match the number of images.")
        if self.fov_masks is not None and len(self.fov_masks) != image_count:
            raise ValueError("Full-image validation FOV masks must match the number of images.")
        if self.batch_size <= 0:
            raise ValueError("Full-image validation batch_size must be greater than zero.")
        if self.num_workers < 0:
            raise ValueError("Full-image validation num_workers must be non-negative.")


@dataclass(frozen=True)
class TrainingResumeState:
    start_epoch: int
    history: dict[str, Any]
    best_selection_value: float
    epochs_without_improvement: int


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    epochs: int | None = None,
    phase: str = "train",
) -> dict[str, float]:
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0

    if epoch is not None and epochs is not None:
        description = f"Epoch {epoch:03d}/{epochs:03d} [{phase}]"
    else:
        description = phase.capitalize()

    progress = tqdm(
        dataloader,
        total=len(dataloader),
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )

    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice_coefficient_from_logits(logits.detach(), targets) * batch_size
        total_samples += batch_size
        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            dice=f"{total_dice / total_samples:.4f}",
        )

    progress.close()

    if total_samples == 0:
        raise RuntimeError("Dataloader produced no samples.")

    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
    }


def save_checkpoint(
    output_path: Path,
    model: torch.nn.Module,
    optimization: OptimizationStack,
    epoch: int,
    best_selection_metric: str,
    best_selection_value: float,
    history: dict[str, Any],
    model_config: dict[str, object],
    patch_config: dict[str, int],
    training_config: dict[str, object],
    epochs_without_improvement: int,
) -> None:
    ensure_dir(output_path.parent)
    checkpoint = {
        "epoch": epoch,
        "best_selection_metric": best_selection_metric,
        "best_selection_value": best_selection_value,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "model_config": model_config,
        "patch_config": patch_config,
        "training_config": training_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimization.optimizer.state_dict(),
    }
    scheduler_state_dict = optimization.scheduler_state_dict()
    if scheduler_state_dict is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state_dict
    torch.save(checkpoint, output_path)


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimization: OptimizationStack,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int,
    patience: int,
    validation_config: ValidationConfig,
    full_image_validation_data: FullImageValidationData | None,
    checkpoint_path: Path,
    last_checkpoint_path: Path,
    model_config: dict[str, object],
    patch_config: dict[str, int],
    training_config: dict[str, object],
    resume_state: TrainingResumeState | None = None,
) -> dict[str, Any]:
    if full_image_validation_data is not None:
        full_image_validation_data.validate()

    if resume_state is None:
        history: dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "train_dice": [],
            "val_patch_dice": [],
            "val_dice": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
            "val_iou": [],
            "val_accuracy": [],
            "lr": [],
            "selection_metric": validation_config.selection_metric,
            "selection_metric_source": validation_config.selection_metric_source,
            "best_checkpoint_selection_metric": validation_config.selection_metric,
            "best_checkpoint_selection_source": validation_config.selection_metric_source,
            "val_threshold": validation_config.val_threshold,
            "run_full_image_validation": validation_config.run_full_image_validation,
            "full_image_validation_every": validation_config.full_image_validation_every,
            "best_epoch": None,
            "best_selection_value": None,
        }
        best_selection_value = _initial_selection_value(validation_config.selection_metric)
        epochs_without_improvement = 0
        start_epoch = 1
    else:
        history = resume_state.history
        history.setdefault(
            "full_image_validation_every",
            validation_config.full_image_validation_every,
        )
        best_selection_value = resume_state.best_selection_value
        epochs_without_improvement = resume_state.epochs_without_improvement
        start_epoch = resume_state.start_epoch

    if start_epoch > epochs:
        raise ValueError(
            f"start_epoch ({start_epoch}) cannot be greater than total epochs ({epochs})."
        )

    for epoch in range(start_epoch, epochs + 1):
        epoch_lr = optimization.current_lr()
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimization.optimizer,
            epoch=epoch,
            epochs=epochs,
            phase="train",
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            epoch=epoch,
            epochs=epochs,
            phase="val",
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_patch_dice"].append(val_metrics["dice"])
        history["lr"].append(epoch_lr)

        full_image_metrics = None
        if (
            validation_config.run_full_image_validation
            and _should_run_full_image_validation_for_epoch(epoch, validation_config)
        ):
            if full_image_validation_data is None:
                raise ValueError("full_image_validation_data is required when full-image validation is enabled.")
            full_image_metrics = run_full_image_validation(
                model=model,
                validation_data=full_image_validation_data,
                device=device,
                threshold=validation_config.val_threshold,
                epoch=epoch,
                epochs=epochs,
            )
        _append_full_image_metrics(history, full_image_metrics)

        optimization.step_scheduler(val_metrics["loss"])

        selection_value = _resolve_selection_value(
            validation_config.selection_metric,
            patch_val_metrics=val_metrics,
            full_image_metrics=full_image_metrics,
        )
        improved = False
        if selection_value is not None:
            improved = _is_improved_selection_metric(
                validation_config.selection_metric,
                current_value=selection_value,
                best_value=best_selection_value,
            )
            if improved:
                best_selection_value = selection_value
                epochs_without_improvement = 0
                history["best_epoch"] = epoch
                history["best_selection_value"] = selection_value
            else:
                epochs_without_improvement += 1
        else:
            history.setdefault("skipped_full_image_validation_epochs", []).append(epoch)

        save_checkpoint(
            output_path=last_checkpoint_path,
            model=model,
            optimization=optimization,
            epoch=epoch,
            best_selection_metric=validation_config.selection_metric,
            best_selection_value=best_selection_value,
            history=history,
            model_config=model_config,
            patch_config=patch_config,
            training_config=training_config,
            epochs_without_improvement=epochs_without_improvement,
        )

        if improved:
            save_checkpoint(
                output_path=checkpoint_path,
                model=model,
                optimization=optimization,
                epoch=epoch,
                best_selection_metric=validation_config.selection_metric,
                best_selection_value=best_selection_value,
                history=history,
                model_config=model_config,
                patch_config=patch_config,
                training_config=training_config,
                epochs_without_improvement=epochs_without_improvement,
            )

        full_image_summary = ""
        if full_image_metrics is not None:
            full_image_summary = (
                f" val_dice={full_image_metrics['dice']:.4f}"
                f" val_f1={full_image_metrics['f1']:.4f}"
                f" val_precision={full_image_metrics['precision']:.4f}"
                f" val_recall={full_image_metrics['recall']:.4f}"
            )

        if selection_value is None:
            selection_summary = f"selection={validation_config.selection_metric}:skipped"
        else:
            selection_summary = (
                f"selection={validation_config.selection_metric}:{selection_value:.4f}"
            )

        tqdm.write(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_patch_dice={val_metrics['dice']:.4f}"
            f"{full_image_summary} "
            f"{selection_summary} "
            f"lr={epoch_lr:.2e}"
        )

        if selection_value is not None and epochs_without_improvement >= patience:
            tqdm.write(f"Early stopping after {epoch} epochs.")
            break

    return history


def predict_patch_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    description: str = "Predict patches",
    show_progress: bool = True,
    use_amp: bool = False,
) -> np.ndarray:
    model.eval()
    probability_batches: list[np.ndarray] = []

    with torch.no_grad():
        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc=description,
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for inputs in progress:
            inputs = inputs.to(device, non_blocking=True)
            with _autocast_context(device=device, use_amp=use_amp):
                logits = model(inputs)
                probabilities = torch.sigmoid(logits)
            probabilities = probabilities.float().cpu().numpy()
            probability_batches.append(probabilities)
        progress.close()

    if not probability_batches:
        raise RuntimeError("No patches were generated for inference.")

    return np.concatenate(probability_batches, axis=0)


def predict_full_image(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
    description: str = "Predict patches",
    show_progress: bool = True,
    use_amp: bool = False,
) -> np.ndarray:
    padded_image, original_shape = pad_image_for_patching(image, patch_size, stride)
    patches = extract_ordered_patches(padded_image, patch_size, stride)

    dataloader = DataLoader(
        PatchArrayDataset(patches),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    patch_probabilities = predict_patch_loader(
        model,
        dataloader,
        device,
        description=description,
        show_progress=show_progress,
        use_amp=use_amp,
    )
    probability_map = reconstruct_from_ordered_patches(
        patches=patch_probabilities,
        output_shape=padded_image.shape[-2:],
        patch_size=patch_size,
        stride=stride,
    )
    return probability_map[:, : original_shape[0], : original_shape[1]]


def run_full_image_validation(
    model: torch.nn.Module,
    validation_data: FullImageValidationData,
    device: torch.device,
    threshold: float,
    epoch: int | None = None,
    epochs: int | None = None,
) -> dict[str, float]:
    validation_data.validate()
    if epoch is not None and epochs is not None:
        description = f"Epoch {epoch:03d}/{epochs:03d} [val_images]"
    else:
        description = "Full-image validation"

    probability_maps: list[np.ndarray] = []
    progress = tqdm(
        range(len(validation_data.images)),
        total=len(validation_data.images),
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )
    for index in progress:
        probability_map = predict_full_image(
            model=model,
            image=validation_data.images[index],
            patch_size=validation_data.patch_size,
            stride=validation_data.stride,
            device=device,
            batch_size=validation_data.batch_size,
            num_workers=validation_data.num_workers,
            description="Validation patches",
            show_progress=False,
        )
        probability_maps.append(probability_map)
    progress.close()

    return compute_binary_segmentation_metrics(
        probability_maps=probability_maps,
        truth_masks=validation_data.masks,
        fov_masks=validation_data.fov_masks,
        threshold=threshold,
    )


def _append_full_image_metrics(
    history: dict[str, Any],
    full_image_metrics: dict[str, float] | None,
) -> None:
    metric_keys = (
        "val_dice",
        "val_f1",
        "val_precision",
        "val_recall",
        "val_iou",
        "val_accuracy",
    )
    if full_image_metrics is None:
        for key in metric_keys:
            history[key].append(None)
        return

    history["val_dice"].append(full_image_metrics["dice"])
    history["val_f1"].append(full_image_metrics["f1"])
    history["val_precision"].append(full_image_metrics["precision"])
    history["val_recall"].append(full_image_metrics["recall"])
    history["val_iou"].append(full_image_metrics["iou"])
    history["val_accuracy"].append(full_image_metrics["accuracy"])


def _initial_selection_value(selection_metric: str) -> float:
    if selection_metric == "val_loss":
        return float("inf")
    return float("-inf")


def _should_run_full_image_validation_for_epoch(
    epoch: int,
    validation_config: ValidationConfig,
) -> bool:
    return epoch % validation_config.full_image_validation_every == 0


def _resolve_selection_value(
    selection_metric: str,
    patch_val_metrics: dict[str, float],
    full_image_metrics: dict[str, float] | None,
) -> float | None:
    if selection_metric == "val_loss":
        return patch_val_metrics["loss"]
    if full_image_metrics is None:
        return None
    metric_map = {
        "val_dice": "dice",
        "val_f1": "f1",
    }
    return full_image_metrics[metric_map[selection_metric]]


def _is_improved_selection_metric(
    selection_metric: str,
    current_value: float,
    best_value: float,
) -> bool:
    if selection_metric == "val_loss":
        return current_value < best_value
    return current_value > best_value
