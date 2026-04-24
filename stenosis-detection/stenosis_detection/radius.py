from __future__ import annotations

from collections.abc import Mapping

import numpy as np


ANGLE_SAMPLES = np.arange(0.0, (2.0 * np.pi) + (np.pi / 1800.0), np.pi / 1800.0, dtype=np.float64)
COSINE_SAMPLES = np.cos(ANGLE_SAMPLES)
SINE_SAMPLES = np.sin(ANGLE_SAMPLES)


def _radius_steps(search_radius: float) -> np.ndarray:
    return np.arange(1.0, float(search_radius) + 0.5, 0.5, dtype=np.float64)


def MoMforSeg1(row_center: int, col_center: int, search_radius: float, image: np.ndarray) -> float:
    rows, cols = image.shape[:2]

    for radius in _radius_steps(search_radius):
        row_positions = row_center + radius * COSINE_SAMPLES
        col_positions = col_center + radius * SINE_SAMPLES
        row_indices = np.rint(row_positions).astype(np.int32)
        col_indices = np.rint(col_positions).astype(np.int32)

        valid = (
            (row_indices >= 1)
            & (row_indices <= rows)
            & (col_indices >= 1)
            & (col_indices <= cols)
        )

        if not np.any(valid):
            continue

        sampled_values = image[row_indices[valid] - 1, col_indices[valid] - 1]
        if np.any(sampled_values != 255):
            return abs(float(radius))

    return 100.0


def build_point_data(
    skeleton_points_rc: np.ndarray,
    mask_gray: np.ndarray,
    search_radius: float,
) -> dict[tuple[int, int], float]:
    point_data: dict[tuple[int, int], float] = {}

    for row, col in skeleton_points_rc:
        row_int = int(row)
        col_int = int(col)
        point_data[(row_int, col_int)] = MoMforSeg1(row_int, col_int, search_radius, mask_gray)

    return point_data


def get_radius(point_data: Mapping[tuple[int, int], float], point: np.ndarray | tuple[int, int]) -> float:
    point_key = (int(point[0]), int(point[1]))
    return float(point_data.get(point_key, np.nan))
