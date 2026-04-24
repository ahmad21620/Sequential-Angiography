from __future__ import annotations

import numpy as np


EIGHT_CONNECTED_DIRECTIONS = np.array(
    [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ],
    dtype=np.int32,
)


def _empty_neighbors() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.int32)


def check_neighbors(binary_mask: np.ndarray, row: int, col: int) -> np.ndarray:
    rows, cols = binary_mask.shape
    neighbors: list[tuple[int, int]] = []

    for delta_row, delta_col in EIGHT_CONNECTED_DIRECTIONS:
        neighbor_row = row + int(delta_row)
        neighbor_col = col + int(delta_col)

        if 1 <= neighbor_row <= rows and 1 <= neighbor_col <= cols and binary_mask[neighbor_row - 1, neighbor_col - 1]:
            neighbors.append((neighbor_row, neighbor_col))

    if not neighbors:
        return _empty_neighbors()

    return np.asarray(neighbors, dtype=np.int32)


def get_neighbors(binary_mask: np.ndarray, current: np.ndarray) -> np.ndarray:
    row = int(current[0])
    col = int(current[1])
    return check_neighbors(binary_mask, row, col)
