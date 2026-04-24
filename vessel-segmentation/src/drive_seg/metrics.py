from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def dice_coefficient_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return float(dice.mean().item())


def flatten_inside_fov(
    probability_maps: Sequence[np.ndarray] | np.ndarray,
    truth_masks: Sequence[np.ndarray] | np.ndarray,
    fov_masks: Sequence[np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(probability_maps) != len(truth_masks) or len(fov_masks) != len(probability_maps):
        raise ValueError("probability_maps, truth_masks, and fov_masks must have the same length.")

    y_scores: list[np.ndarray] = []
    y_true_values: list[np.ndarray] = []
    for probability_map, truth_mask, fov_mask in zip(probability_maps, truth_masks, fov_masks):
        probability_2d = _to_2d_array(probability_map)
        truth_2d = _to_2d_array(truth_mask)
        fov_2d = _to_2d_array(fov_mask) > 0
        if probability_2d.shape != truth_2d.shape or probability_2d.shape != fov_2d.shape:
            raise ValueError(
                "Probability maps, truth masks, and FOV masks must have matching 2D shapes."
            )
        y_scores.append(probability_2d[fov_2d].astype(np.float32, copy=False))
        y_true_values.append(truth_2d[fov_2d].astype(np.uint8, copy=False))

    if not y_scores:
        raise ValueError("At least one probability map is required to compute metrics.")

    return np.concatenate(y_scores), np.concatenate(y_true_values)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _to_2d_array(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim == 3:
        if result.shape[0] != 1:
            raise ValueError(f"Expected a single-channel array, got shape {result.shape}.")
        result = result[0]
    elif result.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D array, got shape {result.shape}.")
    return result


def _build_binary_metrics_from_counts(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    threshold: float,
) -> dict[str, float]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * tp, (2 * tp) + fp + fn)
    return {
        "dice": f1,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "iou": _safe_divide(tp, tp + fp + fn),
        "accuracy": _safe_divide(tn + tp, tn + fp + fn + tp),
        "threshold": threshold,
    }


def compute_binary_segmentation_metrics(
    probability_maps: Sequence[np.ndarray] | np.ndarray,
    truth_masks: Sequence[np.ndarray] | np.ndarray,
    fov_masks: Sequence[np.ndarray] | np.ndarray | None,
    threshold: float = 0.5,
) -> dict[str, float]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in the range [0, 1].")

    if len(probability_maps) != len(truth_masks):
        raise ValueError("probability_maps and truth_masks must have the same length.")
    if fov_masks is not None and len(fov_masks) != len(probability_maps):
        raise ValueError("fov_masks must have the same length as probability_maps.")

    tn = fp = fn = tp = 0
    for index, (probability_map, truth_mask) in enumerate(zip(probability_maps, truth_masks)):
        probability_2d = _to_2d_array(probability_map)
        truth_2d = _to_2d_array(truth_mask) > 0
        if fov_masks is None:
            valid_mask = np.ones_like(truth_2d, dtype=bool)
        else:
            valid_mask = _to_2d_array(fov_masks[index]) > 0

        if probability_2d.shape != truth_2d.shape or valid_mask.shape != truth_2d.shape:
            raise ValueError(
                "Probability maps, truth masks, and FOV masks must have matching 2D shapes."
            )

        predicted_mask = probability_2d >= threshold
        true_negative = (~predicted_mask & ~truth_2d & valid_mask).sum()
        false_positive = (predicted_mask & ~truth_2d & valid_mask).sum()
        false_negative = (~predicted_mask & truth_2d & valid_mask).sum()
        true_positive = (predicted_mask & truth_2d & valid_mask).sum()

        tn += int(true_negative)
        fp += int(false_positive)
        fn += int(false_negative)
        tp += int(true_positive)

    return _build_binary_metrics_from_counts(tn, fp, fn, tp, threshold)


def compute_segmentation_metrics(
    probability_maps: Sequence[np.ndarray] | np.ndarray,
    truth_masks: Sequence[np.ndarray] | np.ndarray,
    fov_masks: Sequence[np.ndarray] | np.ndarray,
    threshold: float = 0.5,
) -> tuple[dict[str, float | list[list[int]]], dict[str, np.ndarray]]:
    y_score, y_true = flatten_inside_fov(probability_maps, truth_masks, fov_masks)
    y_pred = (y_score >= threshold).astype(np.uint8)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()

    metrics: dict[str, float | list[list[int]]] = {
        "auc": auc,
        **_build_binary_metrics_from_counts(tn, fp, fn, tp, threshold),
        "specificity": _safe_divide(tn, tn + fp),
        "confusion_matrix": confusion.tolist(),
    }
    roc_data = {"fpr": fpr, "tpr": tpr}
    return metrics, roc_data
