from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FrameLevelResult, FrameRegistration, LesionObservation
from .registration import transform_point_to_reference


EIGHT_CONNECTED_OFFSETS_XY = np.asarray(
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


@dataclass(slots=True)
class _CenterlineComponent:
    component_id: int
    points_xy: np.ndarray
    kind: str
    ordered_point_indices: np.ndarray | None
    cumulative_lengths: np.ndarray | None
    total_length: float | None


def map_observations_to_reference_centerline(
    frame_result: FrameLevelResult,
    reference_frame: FrameLevelResult,
    registration: FrameRegistration,
) -> list[LesionObservation]:
    mapped_observations: list[LesionObservation] = []
    centerline_components = _build_centerline_components(reference_frame.skeleton_points_xy)

    for observation in frame_result.observations:
        registered_point_xy = transform_point_to_reference(observation.point_xy, registration)
        localization = _localize_point_to_centerline(centerline_components, registered_point_xy)

        mapped_observations.append(
            LesionObservation(
                frame_index=observation.frame_index,
                image_name=observation.image_name,
                point_xy=observation.point_xy.copy(),
                degree=float(observation.degree),
                severity=observation.severity,
                registered_point_xy=registered_point_xy,
                reference_centerline_point_xy=localization.reference_point_xy,
                centerline_distance=localization.centerline_distance,
                centerline_component_id=localization.component_id,
                centerline_arc_length=localization.arc_length,
                centerline_position=localization.normalized_position,
                centerline_component_length=localization.component_length,
                centerline_position_status=localization.status,
                centerline_position_reason=localization.reason,
            )
        )

    return mapped_observations


@dataclass(slots=True)
class _CenterlineLocalization:
    reference_point_xy: np.ndarray | None
    centerline_distance: float | None
    component_id: int | None
    arc_length: float | None
    normalized_position: float | None
    component_length: float | None
    status: str
    reason: str | None


def _localize_point_to_centerline(
    centerline_components: list[_CenterlineComponent],
    point_xy: np.ndarray,
) -> _CenterlineLocalization:
    if point_xy.shape != (2,) or not np.all(np.isfinite(point_xy)):
        return _CenterlineLocalization(
            reference_point_xy=None,
            centerline_distance=None,
            component_id=None,
            arc_length=None,
            normalized_position=None,
            component_length=None,
            status="invalid_point",
            reason="Registered point is not a finite two-dimensional coordinate.",
        )

    if not centerline_components:
        return _CenterlineLocalization(
            reference_point_xy=None,
            centerline_distance=None,
            component_id=None,
            arc_length=None,
            normalized_position=None,
            component_length=None,
            status="empty_centerline",
            reason="Reference frame has no skeleton points.",
        )

    best_component: _CenterlineComponent | None = None
    best_nearest_index: int | None = None
    best_distance = float("inf")

    for component in centerline_components:
        distances = np.linalg.norm(component.points_xy.astype(np.float64) - point_xy.astype(np.float64), axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])

        if _is_better_component_match(
            nearest_distance,
            component,
            best_distance=best_distance,
            best_component=best_component,
        ):
            best_component = component
            best_nearest_index = nearest_index
            best_distance = nearest_distance

    if best_component is None or best_nearest_index is None:
        return _CenterlineLocalization(
            reference_point_xy=None,
            centerline_distance=None,
            component_id=None,
            arc_length=None,
            normalized_position=None,
            component_length=None,
            status="empty_centerline",
            reason="Reference frame has no usable skeleton components.",
        )

    reference_point_xy = best_component.points_xy[best_nearest_index].astype(np.int32)

    if best_component.kind == "single_point":
        return _CenterlineLocalization(
            reference_point_xy=reference_point_xy,
            centerline_distance=best_distance,
            component_id=best_component.component_id,
            arc_length=0.0,
            normalized_position=0.0,
            component_length=0.0,
            status="single_point_component",
            reason="Matched a one-point centerline component; normalized position is fixed at 0.0.",
        )

    if best_component.kind != "path":
        return _CenterlineLocalization(
            reference_point_xy=reference_point_xy,
            centerline_distance=best_distance,
            component_id=best_component.component_id,
            arc_length=None,
            normalized_position=None,
            component_length=best_component.total_length,
            status=f"{best_component.kind}_component",
            reason=_component_kind_reason(best_component.kind),
        )

    ordered_point_indices = best_component.ordered_point_indices
    cumulative_lengths = best_component.cumulative_lengths
    if ordered_point_indices is None or cumulative_lengths is None or best_component.total_length is None:
        return _CenterlineLocalization(
            reference_point_xy=reference_point_xy,
            centerline_distance=best_distance,
            component_id=best_component.component_id,
            arc_length=None,
            normalized_position=None,
            component_length=None,
            status="path_component_invalid",
            reason="Matched a path-like component, but its arc-length representation is unavailable.",
        )

    ordered_match_index = int(np.where(ordered_point_indices == best_nearest_index)[0][0])
    arc_length = float(cumulative_lengths[ordered_match_index])
    component_length = float(best_component.total_length)
    if component_length <= 0.0:
        normalized_position = 0.0
    else:
        normalized_position = arc_length / component_length

    return _CenterlineLocalization(
        reference_point_xy=reference_point_xy,
        centerline_distance=best_distance,
        component_id=best_component.component_id,
        arc_length=arc_length,
        normalized_position=float(np.clip(normalized_position, 0.0, 1.0)),
        component_length=component_length,
        status="ok",
        reason=None,
    )


def _build_centerline_components(skeleton_points_xy: np.ndarray) -> list[_CenterlineComponent]:
    if skeleton_points_xy.size == 0:
        return []

    points_xy = np.asarray(skeleton_points_xy, dtype=np.int32)
    point_lookup = {tuple(int(value) for value in point): index for index, point in enumerate(points_xy)}
    adjacency = _build_adjacency(points_xy, point_lookup)
    component_indices = _connected_components(adjacency)

    components: list[_CenterlineComponent] = []
    for component_id, point_indices in enumerate(component_indices, start=1):
        component_points_xy = points_xy[point_indices].astype(np.int32)
        local_index_by_global_index = {global_index: local_index for local_index, global_index in enumerate(point_indices)}
        component_adjacency = [
            [local_index_by_global_index[neighbor_index] for neighbor_index in adjacency[global_index] if neighbor_index in local_index_by_global_index]
            for global_index in point_indices
        ]
        kind, ordered_point_indices, cumulative_lengths, total_length = _analyze_component(component_points_xy, component_adjacency)
        components.append(
            _CenterlineComponent(
                component_id=component_id,
                points_xy=component_points_xy,
                kind=kind,
                ordered_point_indices=ordered_point_indices,
                cumulative_lengths=cumulative_lengths,
                total_length=total_length,
            )
        )

    return components


def _build_adjacency(
    points_xy: np.ndarray,
    point_lookup: dict[tuple[int, int], int],
) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(len(points_xy))]

    for index, point_xy in enumerate(points_xy):
        x = int(point_xy[0])
        y = int(point_xy[1])

        for delta_x, delta_y in EIGHT_CONNECTED_OFFSETS_XY:
            neighbor_index = point_lookup.get((x + int(delta_x), y + int(delta_y)))
            if neighbor_index is not None:
                adjacency[index].append(int(neighbor_index))

    return adjacency


def _connected_components(adjacency: list[list[int]]) -> list[list[int]]:
    visited = np.zeros((len(adjacency),), dtype=bool)
    components: list[list[int]] = []

    for start_index in range(len(adjacency)):
        if visited[start_index]:
            continue

        stack = [start_index]
        visited[start_index] = True
        component: list[int] = []

        while stack:
            current_index = stack.pop()
            component.append(current_index)

            for neighbor_index in adjacency[current_index]:
                if visited[neighbor_index]:
                    continue
                visited[neighbor_index] = True
                stack.append(neighbor_index)

        components.append(sorted(component))

    return components


def _analyze_component(
    component_points_xy: np.ndarray,
    component_adjacency: list[list[int]],
) -> tuple[str, np.ndarray | None, np.ndarray | None, float | None]:
    if len(component_points_xy) == 1:
        return "single_point", np.asarray([0], dtype=np.int32), np.asarray([0.0], dtype=np.float64), 0.0

    degrees = np.asarray([len(neighbors) for neighbors in component_adjacency], dtype=np.int32)
    if np.any(degrees > 2):
        return "branched", None, None, None

    endpoint_indices = [index for index, degree in enumerate(degrees) if degree == 1]
    if len(endpoint_indices) != 2:
        return "loop", None, None, None

    ordered_point_indices = _order_path_component(component_points_xy, component_adjacency, endpoint_indices)
    if ordered_point_indices is None:
        return "branched", None, None, None

    cumulative_lengths = _compute_cumulative_lengths(component_points_xy[ordered_point_indices])
    total_length = float(cumulative_lengths[-1]) if len(cumulative_lengths) > 0 else 0.0
    return "path", ordered_point_indices, cumulative_lengths, total_length


def _order_path_component(
    component_points_xy: np.ndarray,
    component_adjacency: list[list[int]],
    endpoint_indices: list[int],
) -> np.ndarray | None:
    start_index = min(endpoint_indices, key=lambda index: (int(component_points_xy[index, 0]), int(component_points_xy[index, 1])))
    ordered_indices: list[int] = []
    previous_index: int | None = None
    current_index = start_index

    while True:
        ordered_indices.append(current_index)
        next_candidates = [neighbor_index for neighbor_index in component_adjacency[current_index] if neighbor_index != previous_index]
        if not next_candidates:
            break
        if len(next_candidates) > 1:
            return None

        previous_index = current_index
        current_index = next_candidates[0]

    if len(ordered_indices) != len(component_points_xy):
        return None

    return np.asarray(ordered_indices, dtype=np.int32)


def _compute_cumulative_lengths(ordered_points_xy: np.ndarray) -> np.ndarray:
    if len(ordered_points_xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(ordered_points_xy) == 1:
        return np.asarray([0.0], dtype=np.float64)

    deltas = np.diff(ordered_points_xy.astype(np.float64), axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate((np.asarray([0.0], dtype=np.float64), np.cumsum(segment_lengths, dtype=np.float64)))


def _is_better_component_match(
    distance: float,
    component: _CenterlineComponent,
    *,
    best_distance: float,
    best_component: _CenterlineComponent | None,
) -> bool:
    if best_component is None:
        return True
    if distance < best_distance:
        return True
    if not np.isclose(distance, best_distance):
        return False
    if len(component.points_xy) > len(best_component.points_xy):
        return True
    if len(component.points_xy) < len(best_component.points_xy):
        return False
    return component.component_id < best_component.component_id


def _component_kind_reason(component_kind: str) -> str:
    if component_kind == "branched":
        return "Matched skeleton component contains branches, so a unique normalized arc-length position is unavailable."
    if component_kind == "loop":
        return "Matched skeleton component is loop-like, so endpoint-based normalized arc-length is unavailable."
    return "Centerline-relative position is unavailable for the matched component."
