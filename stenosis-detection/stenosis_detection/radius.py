from __future__ import annotations

from collections.abc import Mapping

import numpy as np


ANGLE_SAMPLES = np.arange(0.0, (2.0 * np.pi) + (np.pi / 1800.0), np.pi / 1800.0, dtype=np.float64)
COSINE_SAMPLES = np.cos(ANGLE_SAMPLES)
SINE_SAMPLES = np.sin(ANGLE_SAMPLES)


def _radius_steps(search_radius: float) -> np.ndarray:
    return np.arange(1.0, float(search_radius) + 0.5, 0.5, dtype=np.float64)


def MoMforSeg1(
    row_center: int,
    col_center: int,
    search_radius: float,
    image: np.ndarray,
    vessel_threshold: int = 127,
    outside_fraction_threshold: float = 0.05,
    min_outside_samples: int = 3,
) -> float:
    rows, cols = image.shape[:2]
    flat_image = image.reshape(-1)
    required_outside_fraction = max(0.0, float(outside_fraction_threshold))
    required_outside_samples = max(1, int(min_outside_samples))

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

        sampled_positions = ((row_indices[valid] - 1) * cols) + (col_indices[valid] - 1)
        sampled_values = flat_image[np.unique(sampled_positions)]
        outside_count = int(np.count_nonzero(sampled_values < vessel_threshold))
        outside_fraction = outside_count / float(len(sampled_values))

        if outside_count >= required_outside_samples and outside_fraction >= required_outside_fraction:
            return abs(float(radius))

    return 100.0


def build_point_data(
    skeleton_points_rc: np.ndarray,
    mask_gray: np.ndarray,
    search_radius: float,
    vessel_threshold: int = 127,
    outside_fraction_threshold: float = 0.05,
    min_outside_samples: int = 3,
) -> dict[tuple[int, int], float]:
    point_data: dict[tuple[int, int], float] = {}

    for row, col in skeleton_points_rc:
        row_int = int(row)
        col_int = int(col)
        point_data[(row_int, col_int)] = MoMforSeg1(
            row_int,
            col_int,
            search_radius,
            mask_gray,
            vessel_threshold,
            outside_fraction_threshold,
            min_outside_samples,
        )

    return point_data


def get_radius(point_data: Mapping[tuple[int, int], float], point: np.ndarray | tuple[int, int]) -> float:
    point_key = (int(point[0]), int(point[1]))
    return float(point_data.get(point_key, np.nan))
