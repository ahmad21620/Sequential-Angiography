from __future__ import annotations

from typing import Mapping

import numpy as np

from .radius import get_radius


def duilie(
    start_index: int,
    shortest_path: np.ndarray,
    queue: list[tuple[int, int]],
    point_data: Mapping[tuple[int, int], float],
) -> list[tuple[int, int]]:
    if start_index >= len(shortest_path) - 1:
        return queue

    i = start_index

    # Find the first decreasing point and enqueue it.
    while i <= len(shortest_path) - 1:
        current_point = shortest_path[i - 1]
        next_point = shortest_path[i]
        current_radius = get_radius(point_data, current_point)
        next_radius = get_radius(point_data, next_point)

        if current_radius > next_radius:
            queue.append((int(current_point[0]), int(current_point[1])))
            break

        i += 1

    # Find the last point in the decreasing run.
    while i <= len(shortest_path) - 1:
        current_point = shortest_path[i - 1]
        next_point = shortest_path[i]
        current_radius = get_radius(point_data, current_point)
        next_radius = get_radius(point_data, next_point)

        if current_radius >= next_radius:
            i += 1
        else:
            queue.append((int(current_point[0]), int(current_point[1])))
            break

    # Find the last point in the increasing run.
    while i <= len(shortest_path) - 1:
        current_point = shortest_path[i - 1]
        next_point = shortest_path[i]
        current_radius = get_radius(point_data, current_point)
        next_radius = get_radius(point_data, next_point)

        if current_radius <= next_radius:
            i += 1
        else:
            queue.append((int(current_point[0]), int(current_point[1])))
            break

    return duilie(i, shortest_path, queue, point_data)


def collect_queue(shortest_path: np.ndarray, point_data: Mapping[tuple[int, int], float]) -> np.ndarray:
    queue = duilie(1, shortest_path, [], point_data)
    if not queue:
        return np.zeros((0, 2), dtype=np.int32)

    return np.asarray(queue, dtype=np.int32)
