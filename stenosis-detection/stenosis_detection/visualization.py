from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .pipeline import StenosisDetectionResult

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - exercised only when matplotlib is absent.
    plt = None


def save_detection_outputs(
    result: StenosisDetectionResult,
    output_dir: str | Path,
    *,
    show: bool = False,
    file_prefix: str | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    centerline_image = create_centerline_visualization(result.mask_gray, result.skeleton_points_xy)
    segmentation_image = create_segmentation_visualization(
        result.mask_gray,
        result.skeleton_points_xy,
        result.segmentation_points_xy,
        result.filtered_segmentation_points_xy,
    )
    stenosis_mask_image = create_stenosis_visualization(
        result.mask_gray,
        result.stenosis_points_xy,
        result.stenosis_degrees,
    )
    stenosis_original_image = create_stenosis_visualization(
        result.original_image_bgr,
        result.stenosis_points_xy,
        result.stenosis_degrees,
    )

    output_files = build_output_paths(output_path, file_prefix=file_prefix)

    cv2.imwrite(str(output_files["centerline"]), centerline_image)
    cv2.imwrite(str(output_files["segmentation_points"]), segmentation_image)
    cv2.imwrite(str(output_files["stenosis_mask"]), stenosis_mask_image)
    cv2.imwrite(str(output_files["stenosis_original"]), stenosis_original_image)
    output_files["results_json"].write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    if show:
        _show_outputs(
            centerline_image=centerline_image,
            segmentation_image=segmentation_image,
            stenosis_mask_image=stenosis_mask_image,
            stenosis_original_image=stenosis_original_image,
        )

    return output_files


def build_output_paths(output_dir: str | Path, *, file_prefix: str | None = None) -> dict[str, Path]:
    output_path = Path(output_dir)
    prefix = f"{file_prefix}_" if file_prefix else ""

    return {
        "centerline": output_path / f"{prefix}centerline.png",
        "segmentation_points": output_path / f"{prefix}segmentation_points.png",
        "stenosis_mask": output_path / f"{prefix}stenosis_result_mask.png",
        "stenosis_original": output_path / f"{prefix}stenosis_result_original.png",
        "results_json": output_path / f"{prefix}stenosis_results.json",
    }


def create_centerline_visualization(mask_gray: np.ndarray, skeleton_points_xy: np.ndarray) -> np.ndarray:
    canvas = _to_bgr_canvas(mask_gray)
    _paint_points(canvas, skeleton_points_xy, color=(0, 0, 255))
    return canvas


def create_segmentation_visualization(
    mask_gray: np.ndarray,
    skeleton_points_xy: np.ndarray,
    segmentation_points_xy: np.ndarray,
    filtered_segmentation_points_xy: np.ndarray,
) -> np.ndarray:
    canvas = _to_bgr_canvas(mask_gray)
    _paint_points(canvas, skeleton_points_xy, color=(0, 0, 255))

    for point in _to_zero_based_xy(segmentation_points_xy):
        cv2.circle(canvas, tuple(point), radius=3, color=(255, 0, 255), thickness=-1)

    for point in _to_zero_based_xy(filtered_segmentation_points_xy):
        cv2.circle(canvas, tuple(point), radius=5, color=(255, 255, 0), thickness=1)

    return canvas


def create_stenosis_visualization(base_image: np.ndarray, stenosis_points_xy: np.ndarray, stenosis_degrees: np.ndarray) -> np.ndarray:
    canvas = _to_bgr_canvas(base_image)

    for point, degree in zip(_to_zero_based_xy(stenosis_points_xy), stenosis_degrees, strict=False):
        color = _severity_color(float(degree))
        cv2.circle(canvas, tuple(point), radius=6, color=color, thickness=2)

    return canvas


def _show_outputs(
    *,
    centerline_image: np.ndarray,
    segmentation_image: np.ndarray,
    stenosis_mask_image: np.ndarray,
    stenosis_original_image: np.ndarray,
) -> None:
    if plt is None:
        raise RuntimeError("Interactive display requires matplotlib. Install it or omit --show.")

    figure, axes = plt.subplots(2, 2, figsize=(12, 14))
    axes = axes.ravel()
    items = [
        ("Centerline", centerline_image),
        ("Segmentation Points", segmentation_image),
        ("Stenosis on Mask", stenosis_mask_image),
        ("Stenosis on Original", stenosis_original_image),
    ]

    for axis, (title, image) in zip(axes, items, strict=False):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    plt.show()


def _to_bgr_canvas(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy().astype(np.uint8)

    raise ValueError(f"Unsupported visualization image shape: {image.shape}")


def _to_zero_based_xy(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    return np.clip(points_xy.astype(np.int32) - 1, 0, None)


def _paint_points(canvas: np.ndarray, points_xy: np.ndarray, color: tuple[int, int, int]) -> None:
    zero_based_points = _to_zero_based_xy(points_xy)
    if zero_based_points.size == 0:
        return

    height, width = canvas.shape[:2]
    valid = (
        (zero_based_points[:, 0] >= 0)
        & (zero_based_points[:, 0] < width)
        & (zero_based_points[:, 1] >= 0)
        & (zero_based_points[:, 1] < height)
    )
    valid_points = zero_based_points[valid]

    canvas[valid_points[:, 1], valid_points[:, 0]] = color


def _severity_color(stenosis_degree: float) -> tuple[int, int, int]:
    if stenosis_degree > 0.75:
        return (0, 0, 255)
    if stenosis_degree > 0.5:
        return (0, 255, 0)
    return (255, 0, 0)
