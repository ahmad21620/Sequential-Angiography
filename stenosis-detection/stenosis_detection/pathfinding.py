from __future__ import annotations

from collections import deque

import numpy as np

from .neighbors import get_neighbors


class PathNotFoundError(RuntimeError):
    pass


def findpath2(binary_mask: np.ndarray, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, int]:
    if len(start) != 2 or len(goal) != 2:
        raise ValueError("Start and goal must be two-dimensional points.")

    # Preserve the MATLAB helper behavior: inputs arrive as [x, y] and are
    # immediately flipped into [row, col] before the search.
    start_rc = np.flip(np.asarray(start, dtype=np.int32))
    goal_rc = np.flip(np.asarray(goal, dtype=np.int32))

    rows, cols = binary_mask.shape
    dist = np.full((rows, cols), np.inf, dtype=np.float64)
    visited = np.zeros((rows, cols), dtype=bool)
    previous: dict[tuple[int, int], tuple[int, int]] = {}

    start_key = (int(start_rc[0]), int(start_rc[1]))
    goal_key = (int(goal_rc[0]), int(goal_rc[1]))

    queue: deque[tuple[int, int]] = deque([start_key])
    dist[start_key[0] - 1, start_key[1] - 1] = 0.0
    visited[start_key[0] - 1, start_key[1] - 1] = True

    while queue:
        current_key = queue.popleft()
        current = np.asarray(current_key, dtype=np.int32)
        neighbors = get_neighbors(binary_mask, current)

        for neighbor in neighbors:
            neighbor_key = (int(neighbor[0]), int(neighbor[1]))
            neighbor_row = neighbor_key[0] - 1
            neighbor_col = neighbor_key[1] - 1

            if not visited[neighbor_row, neighbor_col]:
                visited[neighbor_row, neighbor_col] = True
                new_dist = dist[current_key[0] - 1, current_key[1] - 1] + 1.0

                if new_dist < dist[neighbor_row, neighbor_col]:
                    dist[neighbor_row, neighbor_col] = new_dist
                    previous[neighbor_key] = current_key
                    queue.append(neighbor_key)

                    if neighbor_key == goal_key:
                        break

    goal_distance = dist[goal_key[0] - 1, goal_key[1] - 1]
    if np.isinf(goal_distance):
        raise PathNotFoundError("No path found from start to goal.")

    path: list[tuple[int, int]] = [goal_key]
    current_key = goal_key
    while current_key != start_key:
        current_key = previous[current_key]
        path.append(current_key)

    shortest_path = np.flip(np.asarray(path, dtype=np.int32), axis=0)

    # MATLAB returns the edge count, not the number of nodes in the path.
    shortest_path_length = int(goal_distance)
    return shortest_path, shortest_path_length
