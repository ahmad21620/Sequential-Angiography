from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import FrameLevelResult


def load_resized_image(frame_path: str | None, frame_result: FrameLevelResult, imread_flag: int) -> np.ndarray | None:
    resolved_path = resolve_asset_path(frame_path, frame_result)
    if resolved_path is None:
        return None

    image = cv2.imread(str(resolved_path), imread_flag)
    if image is None:
        return None

    target_width = int(frame_result.width or image.shape[1])
    target_height = int(frame_result.height or image.shape[0])
    if image.shape[1] == target_width and image.shape[0] == target_height:
        return image

    interpolation = cv2.INTER_AREA if image.shape[1] > target_width or image.shape[0] > target_height else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def resolve_asset_path(asset_path: str | None, frame_result: FrameLevelResult) -> Path | None:
    if asset_path is None or not asset_path.strip():
        return None

    candidate = Path(asset_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    search_roots = [Path.cwd(), frame_result.result_path.parent, *frame_result.result_path.parent.parents]
    for root in search_roots:
        resolved_candidate = root / candidate
        if resolved_candidate.exists():
            return resolved_candidate

    return None


def paint_points(canvas: np.ndarray, points_xy: np.ndarray, *, color: tuple[int, int, int]) -> None:
    zero_based_points_xy = to_zero_based_xy(points_xy)
    if zero_based_points_xy.size == 0:
        return

    rounded_points_xy = np.rint(zero_based_points_xy).astype(np.int32)
    height, width = canvas.shape[:2]
    valid_mask = (
        (rounded_points_xy[:, 0] >= 0)
        & (rounded_points_xy[:, 0] < width)
        & (rounded_points_xy[:, 1] >= 0)
        & (rounded_points_xy[:, 1] < height)
    )
    valid_points_xy = rounded_points_xy[valid_mask]
    canvas[valid_points_xy[:, 1], valid_points_xy[:, 0]] = color


def to_zero_based_xy(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    return points_xy.astype(np.float64) - 1.0
