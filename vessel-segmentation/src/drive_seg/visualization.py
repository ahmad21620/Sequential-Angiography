from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .utils import ensure_dir


def _grayscale_to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.squeeze(image)
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).astype(np.uint8)


def _to_rgb_panel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    return image


def _probability_to_binary_uint8(
    probability_map: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (np.squeeze(probability_map) >= threshold).astype(np.uint8) * 255


def _plot_history_series(
    axis: plt.Axes,
    history: dict[str, object],
    key: str,
    label: str,
) -> None:
    values = history.get(key)
    if not isinstance(values, list) or not values:
        return
    array = np.asarray([np.nan if value is None else value for value in values], dtype=np.float32)
    if np.isnan(array).all():
        return
    axis.plot(array, label=label)


def save_training_history(history: dict[str, object], output_path: Path) -> None:
    ensure_dir(output_path.parent)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    _plot_history_series(axes[0], history, "train_loss", "train_loss")
    _plot_history_series(axes[0], history, "val_loss", "val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    _plot_history_series(axes[1], history, "train_dice", "train_dice")
    _plot_history_series(axes[1], history, "val_patch_dice", "val_patch_dice")
    _plot_history_series(axes[1], history, "val_dice", "val_dice")
    _plot_history_series(axes[1], history, "val_f1", "val_f1")
    axes[1].set_title("Segmentation Scores")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_value: float,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)

    figure = plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_evaluation_prediction_panels(
    images: list[np.ndarray] | np.ndarray,
    truth_masks: list[np.ndarray] | np.ndarray,
    probability_maps: list[np.ndarray] | np.ndarray,
    image_ids: list[str],
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    for image, truth_mask, probability_map, image_id in zip(
        images,
        truth_masks,
        probability_maps,
        image_ids,
    ):
        panel = np.concatenate(
            [
                _grayscale_to_uint8(image),
                _grayscale_to_uint8(truth_mask),
                _grayscale_to_uint8(probability_map),
            ],
            axis=1,
        )
        Image.fromarray(panel).save(output_dir / f"{image_id}.png")


def save_custom_prediction(
    original_rgb: np.ndarray,
    probability_map: np.ndarray,
    output_dir: Path,
    image_id: Path,
    threshold: float,
) -> None:
    probability_path, binary_path, panel_path = get_custom_prediction_output_paths(
        output_dir=output_dir,
        image_id=image_id,
        create_parent=True,
    )

    rgb_image = np.transpose(original_rgb, (1, 2, 0)).astype(np.uint8)
    probability_u8 = _grayscale_to_uint8(probability_map)
    binary_u8 = _probability_to_binary_uint8(probability_map, threshold)

    Image.fromarray(probability_u8).save(probability_path)
    Image.fromarray(binary_u8).save(binary_path)

    panel = np.concatenate(
        [
            _to_rgb_panel(rgb_image),
            _to_rgb_panel(probability_u8),
            _to_rgb_panel(binary_u8),
        ],
        axis=1,
    )
    Image.fromarray(panel).save(panel_path)


def get_custom_prediction_output_paths(
    output_dir: Path,
    image_id: Path,
    create_parent: bool = False,
) -> tuple[Path, Path, Path]:
    destination_dir = output_dir / image_id.parent
    if create_parent:
        destination_dir = ensure_dir(destination_dir)
    image_stem = image_id.stem
    return (
        destination_dir / f"{image_stem}_probability.png",
        destination_dir / f"{image_stem}_binary.png",
        destination_dir / f"{image_stem}_panel.png",
    )


def custom_prediction_outputs_exist(output_dir: Path, image_id: Path) -> bool:
    probability_path, binary_path, panel_path = get_custom_prediction_output_paths(
        output_dir=output_dir,
        image_id=image_id,
    )
    return probability_path.exists() and binary_path.exists() and panel_path.exists()


def get_stenosis_mask_output_path(
    masks_root: Path,
    image_id: Path,
    create_parent: bool = False,
) -> Path:
    destination_dir = masks_root / image_id.parent
    if create_parent:
        destination_dir = ensure_dir(destination_dir)
    return destination_dir / f"{image_id.stem}_mask.png"


def stenosis_mask_output_exists(masks_root: Path, image_id: Path) -> bool:
    return get_stenosis_mask_output_path(
        masks_root=masks_root,
        image_id=image_id,
    ).exists()


def save_stenosis_mask(
    probability_map: np.ndarray,
    masks_root: Path,
    image_id: Path,
    threshold: float,
) -> Path:
    output_path = get_stenosis_mask_output_path(
        masks_root=masks_root,
        image_id=image_id,
        create_parent=True,
    )
    binary_u8 = _probability_to_binary_uint8(probability_map, threshold)
    Image.fromarray(binary_u8).save(output_path)
    return output_path
