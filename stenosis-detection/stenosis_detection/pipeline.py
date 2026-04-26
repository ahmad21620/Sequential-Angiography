from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .neighbors import check_neighbors
from .pathfinding import PathNotFoundError, findpath2
from .queue_logic import collect_queue
from .radius import build_point_data, get_radius
from .thinning import thin_binary_mask


SLICE_FRAME_PATTERN = re.compile(r"^slice_(\d+)$", re.IGNORECASE)


@dataclass(slots=True)
class PipelineConfig:
    resize_height: int = 800
    resize_width: int = 600
    mask_threshold: int = 127
    min_component_area: int = 16
    remove_border_artifacts: bool = True
    border_margin_px: int = 3
    border_artifact_max_height: int = 12
    border_artifact_min_width_ratio: float = 0.5
    radius_search_range: float = 110.0
    radius_vessel_threshold: int = 127
    radius_outside_fraction_threshold: float = 0.05
    radius_min_outside_samples: int = 3
    segmentation_distance_threshold: float = 8.0
    stenosis_threshold: float = 0.25
    average_radius_threshold: float = 4.0
    final_point_distance_threshold: float = 10.0


@dataclass(slots=True)
class StenosisRecord:
    x: int
    y: int
    degree: float
    severity: str


@dataclass(slots=True)
class StenosisDetectionResult:
    image_path: Path
    mask_path: Path
    config: PipelineConfig
    original_image_bgr: np.ndarray
    original_mask_gray: np.ndarray
    mask_gray: np.ndarray
    binary_mask: np.ndarray
    skeleton_mask: np.ndarray
    skeleton_points_rc: np.ndarray
    skeleton_points_xy: np.ndarray
    point_data: dict[tuple[int, int], float]
    segmentation_points_xy: np.ndarray
    filtered_segmentation_points_xy: np.ndarray
    stenosis_points_xy: np.ndarray
    stenosis_degrees: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "mask_path": str(self.mask_path),
            "frame": self.frame_metadata,
            "coordinate_system": "MATLAB-compatible 1-based x/y pixel coordinates",
            "config": asdict(self.config),
            "counts": {
                "skeleton_points": int(len(self.skeleton_points_xy)),
                "segmentation_points": int(len(self.segmentation_points_xy)),
                "filtered_segmentation_points": int(len(self.filtered_segmentation_points_xy)),
                "stenosis_points": int(len(self.stenosis_points_xy)),
            },
            "skeleton_points": _serialize_points_xy(self.skeleton_points_xy),
            "stenosis_points": [
                {
                    "x": int(record.x),
                    "y": int(record.y),
                    "degree": float(record.degree),
                    "severity": record.severity,
                }
                for record in self.stenosis_records
            ],
        }

    @property
    def frame_metadata(self) -> dict[str, Any]:
        height, width = self.original_image_bgr.shape[:2]
        return {
            "image_name": self.image_path.name,
            "image_stem": self.image_path.stem,
            "frame_index": _extract_frame_index(self.image_path),
            "view_id": _build_view_id(self.image_path),
            "width": int(width),
            "height": int(height),
        }

    @property
    def stenosis_records(self) -> list[StenosisRecord]:
        return [
            StenosisRecord(
                x=int(point[0]),
                y=int(point[1]),
                degree=float(degree),
                severity=_classify_stenosis(float(degree)),
            )
            for point, degree in zip(self.stenosis_points_xy, self.stenosis_degrees, strict=False)
        ]


def run_stenosis_detection(
    image_path: str | Path,
    mask_path: str | Path,
    config: PipelineConfig | None = None,
) -> StenosisDetectionResult:
    pipeline_config = config or PipelineConfig()
    resolved_image_path = Path(image_path)
    resolved_mask_path = Path(mask_path)

    original_image_bgr = _load_original_image(resolved_image_path, pipeline_config)
    original_mask_gray, mask_gray, binary_mask = _load_mask_image_with_debug(resolved_mask_path, pipeline_config)

    skeleton_mask = thin_binary_mask(binary_mask)
    skeleton_points_rc = _skeleton_points_rc(skeleton_mask)
    skeleton_points_xy = _flip_points(skeleton_points_rc)

    point_data = build_point_data(
        skeleton_points_rc,
        mask_gray,
        pipeline_config.radius_search_range,
        vessel_threshold=pipeline_config.radius_vessel_threshold,
        outside_fraction_threshold=pipeline_config.radius_outside_fraction_threshold,
        min_outside_samples=pipeline_config.radius_min_outside_samples,
    )

    segmentation_points_xy = _detect_segmentation_points(skeleton_mask, skeleton_points_rc)
    filtered_segmentation_points_xy = _filter_nearby_segmentation_points(
        segmentation_points_xy,
        pipeline_config.segmentation_distance_threshold,
    )

    raw_stenosis_points_rc: list[tuple[int, int]] = []
    raw_stenosis_degrees: list[float] = []

    for index in range(len(filtered_segmentation_points_xy) - 1):
        start_point = filtered_segmentation_points_xy[index]
        end_point = filtered_segmentation_points_xy[index + 1]

        try:
            shortest_path_rc, shortest_path_length = findpath2(skeleton_mask, start_point, end_point)
        except PathNotFoundError:
            continue

        average_radius = _average_path_radius(shortest_path_rc, shortest_path_length, point_data)
        queue_rc = collect_queue(shortest_path_rc, point_data)
        middle_points_rc, stenosis_degrees = _detect_stenosis_from_queue(
            queue_rc,
            average_radius,
            point_data,
            pipeline_config,
        )

        raw_stenosis_points_rc.extend((int(point[0]), int(point[1])) for point in middle_points_rc)
        raw_stenosis_degrees.extend(float(value) for value in stenosis_degrees)

    stenosis_points_xy, stenosis_degrees = _finalize_stenosis_points(
        raw_stenosis_points_rc,
        raw_stenosis_degrees,
        pipeline_config.final_point_distance_threshold,
    )

    return StenosisDetectionResult(
        image_path=resolved_image_path,
        mask_path=resolved_mask_path,
        config=pipeline_config,
        original_image_bgr=original_image_bgr,
        original_mask_gray=original_mask_gray,
        mask_gray=mask_gray,
        binary_mask=binary_mask,
        skeleton_mask=skeleton_mask,
        skeleton_points_rc=skeleton_points_rc,
        skeleton_points_xy=skeleton_points_xy,
        point_data=point_data,
        segmentation_points_xy=segmentation_points_xy,
        filtered_segmentation_points_xy=filtered_segmentation_points_xy,
        stenosis_points_xy=stenosis_points_xy,
        stenosis_degrees=stenosis_degrees,
    )


def _load_original_image(image_path: Path, config: PipelineConfig) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read original image: {image_path}")

    image = _ensure_uint8(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3:
        image = image[:, :, :3]
    else:
        raise ValueError(f"Unsupported original image shape: {image.shape}")

    return _resize_image(image, config.resize_width, config.resize_height)


def _load_mask_image(mask_path: Path, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    _, mask_gray, binary_mask = _load_mask_image_with_debug(mask_path, config)
    return mask_gray, binary_mask


def _load_mask_image_with_debug(mask_path: Path, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Unable to read vessel mask: {mask_path}")

    mask = _ensure_uint8(mask)

    if mask.ndim == 3 and mask.shape[2] == 4:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR)

    resized_mask = _resize_mask(mask, config.resize_width, config.resize_height)

    if resized_mask.ndim == 3:
        mask_gray = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = resized_mask

    if mask_gray.max(initial=0) <= 1:
        mask_gray = np.clip(mask_gray.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)

    original_mask_gray = mask_gray.copy()
    _, mask_gray = cv2.threshold(mask_gray, float(config.mask_threshold), 255, cv2.THRESH_BINARY)
    binary_mask = _clean_binary_mask(mask_gray > 0, config)
    mask_gray = (binary_mask.astype(np.uint8) * 255).astype(np.uint8)
    return original_mask_gray, mask_gray, binary_mask


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)

    if image.size == 0:
        return image.astype(np.uint8)

    if image.min(initial=0.0) >= 0.0 and image.max(initial=0.0) <= 1.0:
        image = image * 255.0
    elif image.max(initial=0.0) > 255.0 or image.min(initial=0.0) < 0.0:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _clean_binary_mask(binary_mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    cleaned_mask = _remove_small_components(binary_mask, config.min_component_area)

    if config.remove_border_artifacts:
        cleaned_mask = _remove_border_artifact_components(cleaned_mask, config)
        cleaned_mask = _remove_long_horizontal_border_runs(cleaned_mask, config)
        cleaned_mask = _remove_border_band_residue_components(cleaned_mask, config)
        cleaned_mask = _remove_small_components(cleaned_mask, config.min_component_area)

    return np.asarray(cleaned_mask, dtype=bool)


def _remove_small_components(binary_mask: np.ndarray, min_component_area: int) -> np.ndarray:
    min_area = int(min_component_area)
    if min_area <= 1 or not np.any(binary_mask):
        return np.asarray(binary_mask, dtype=bool)

    labels, stats = _connected_component_labels(binary_mask)
    keep_labels = np.ones(len(stats), dtype=bool)
    keep_labels[0] = False
    keep_labels[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area

    return keep_labels[labels]


def _remove_border_artifact_components(binary_mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    if not np.any(binary_mask):
        return np.asarray(binary_mask, dtype=bool)

    image_height, image_width = binary_mask.shape[:2]
    if image_height == 0 or image_width == 0:
        return np.asarray(binary_mask, dtype=bool)

    labels, stats = _connected_component_labels(binary_mask)
    keep_labels = np.ones(len(stats), dtype=bool)
    keep_labels[0] = False

    border_margin_px = max(0, int(config.border_margin_px))
    max_artifact_height = max(0, int(config.border_artifact_max_height))
    min_width_ratio = max(0.0, float(config.border_artifact_min_width_ratio))
    min_artifact_width = int(np.ceil(float(image_width) * min_width_ratio))

    for label in range(1, len(stats)):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        right = left + component_width - 1
        bottom = top + component_height - 1

        touches_border = _component_touches_border(
            left,
            top,
            right,
            bottom,
            image_width,
            image_height,
            border_margin_px,
        )
        is_border_artifact = _is_long_thin_border_artifact(
            component_width,
            component_height,
            component_area,
            max_artifact_height,
            min_artifact_width,
        )

        if touches_border and is_border_artifact:
            keep_labels[label] = False

    return keep_labels[labels]


def _remove_long_horizontal_border_runs(binary_mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    if not np.any(binary_mask):
        return np.asarray(binary_mask, dtype=bool)

    image_height, image_width = binary_mask.shape[:2]
    if image_height == 0 or image_width == 0:
        return np.asarray(binary_mask, dtype=bool)

    artifact_band_px = _border_artifact_band_px(config)
    max_artifact_height = max(0, int(config.border_artifact_max_height))
    min_width_ratio = max(0.0, float(config.border_artifact_min_width_ratio))
    min_artifact_width = int(np.ceil(float(image_width) * min_width_ratio))
    if min_artifact_width <= 0 or max_artifact_height <= 0:
        return np.asarray(binary_mask, dtype=bool)

    cleaned_mask = np.asarray(binary_mask, dtype=bool).copy()
    border_rows = _border_band_indices(image_height, artifact_band_px)
    for row_index in border_rows:
        row = cleaned_mask[row_index]
        for start, stop in _horizontal_artifact_candidates(row, min_artifact_width):
            if _horizontal_span_thickness(cleaned_mask, row_index, start, stop) <= max_artifact_height:
                row[start:stop] = False

    return cleaned_mask


def _remove_border_band_residue_components(binary_mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    if not np.any(binary_mask):
        return np.asarray(binary_mask, dtype=bool)

    image_height, image_width = binary_mask.shape[:2]
    if image_height == 0 or image_width == 0:
        return np.asarray(binary_mask, dtype=bool)

    artifact_band_px = _border_artifact_band_px(config)
    labels, stats = _connected_component_labels(binary_mask)
    keep_labels = np.ones(len(stats), dtype=bool)
    keep_labels[0] = False

    for label in range(1, len(stats)):
        top = int(stats[label, cv2.CC_STAT_TOP])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        bottom = top + component_height - 1

        if _component_is_contained_in_border_band(
            top,
            bottom,
            image_height,
            artifact_band_px,
        ):
            keep_labels[label] = False

    return keep_labels[labels]


def _border_artifact_band_px(config: PipelineConfig) -> int:
    return max(0, int(config.border_margin_px), int(config.border_artifact_max_height))


def _border_band_indices(image_size: int, border_margin_px: int) -> np.ndarray:
    border_width = min(image_size, border_margin_px + 1)
    if border_width <= 0:
        return np.zeros((0,), dtype=np.int32)

    top_indices = np.arange(0, border_width, dtype=np.int32)
    bottom_start = max(0, image_size - border_width)
    bottom_indices = np.arange(bottom_start, image_size, dtype=np.int32)
    return np.unique(np.concatenate((top_indices, bottom_indices))).astype(np.int32)


def _true_runs(row: np.ndarray) -> list[tuple[int, int]]:
    row_values = np.asarray(row, dtype=bool)
    if row_values.size == 0 or not np.any(row_values):
        return []

    padded = np.concatenate(([False], row_values, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0]
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def _horizontal_artifact_candidates(row: np.ndarray, min_artifact_width: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for start, stop in _true_runs(row):
        if stop - start >= min_artifact_width:
            candidates.append((start, stop))

    true_columns = np.where(row)[0]
    if true_columns.size >= min_artifact_width:
        span_start = int(true_columns[0])
        span_stop = int(true_columns[-1]) + 1
        if span_stop - span_start >= min_artifact_width:
            candidates.append((span_start, span_stop))

    return _deduplicate_runs(candidates)


def _deduplicate_runs(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not runs:
        return []

    unique_runs = sorted(set(runs))
    merged_runs: list[tuple[int, int]] = []
    for start, stop in unique_runs:
        if not merged_runs or start > merged_runs[-1][1]:
            merged_runs.append((start, stop))
        else:
            previous_start, previous_stop = merged_runs[-1]
            merged_runs[-1] = (previous_start, max(previous_stop, stop))

    return merged_runs


def _horizontal_span_thickness(binary_mask: np.ndarray, row_index: int, start: int, stop: int) -> int:
    span_width = max(1, stop - start)
    row_fill_threshold = max(1, int(np.ceil(float(span_width) * 0.5)))
    thickness = 1

    previous_row = row_index - 1
    while previous_row >= 0 and np.count_nonzero(binary_mask[previous_row, start:stop]) >= row_fill_threshold:
        thickness += 1
        previous_row -= 1

    next_row = row_index + 1
    while next_row < binary_mask.shape[0] and np.count_nonzero(binary_mask[next_row, start:stop]) >= row_fill_threshold:
        thickness += 1
        next_row += 1

    return thickness


def _component_touches_border(
    left: int,
    top: int,
    right: int,
    bottom: int,
    image_width: int,
    image_height: int,
    border_margin_px: int,
) -> bool:
    return (
        left <= border_margin_px
        or top <= border_margin_px
        or right >= image_width - 1 - border_margin_px
        or bottom >= image_height - 1 - border_margin_px
    )


def _component_is_contained_in_border_band(
    top: int,
    bottom: int,
    image_height: int,
    artifact_band_px: int,
) -> bool:
    if artifact_band_px <= 0:
        return False

    bottom_band_start = max(0, image_height - artifact_band_px)
    return (
        bottom < artifact_band_px
        or top >= bottom_band_start
    )


def _is_long_thin_border_artifact(
    component_width: int,
    component_height: int,
    component_area: int,
    max_artifact_height: int,
    min_artifact_width: int,
) -> bool:
    if component_width <= 0:
        return False

    effective_height = float(component_area) / float(component_width)
    return (
        (component_height <= max_artifact_height or effective_height <= float(max_artifact_height))
        and component_width > component_height
        and component_width >= min_artifact_width
    )


def _connected_component_labels(binary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(binary_mask, dtype=np.uint8),
        connectivity=8,
    )
    return labels, stats


def _skeleton_points_rc(skeleton_mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(skeleton_mask)
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    return np.column_stack((rows + 1, cols + 1)).astype(np.int32)


def _flip_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    return np.fliplr(points).astype(np.int32)


def _detect_segmentation_points(skeleton_mask: np.ndarray, skeleton_points_rc: np.ndarray) -> np.ndarray:
    segmentation_points: list[tuple[int, int]] = []

    for row, col in skeleton_points_rc:
        neighbors = check_neighbors(skeleton_mask, int(row), int(col))
        if len(neighbors) == 3:
            segmentation_points.append((int(col), int(row)))

    if not segmentation_points:
        return np.zeros((0, 2), dtype=np.int32)

    return np.asarray(segmentation_points, dtype=np.int32)


def _filter_nearby_segmentation_points(segmentation_points_xy: np.ndarray, distance_threshold: float) -> np.ndarray:
    if len(segmentation_points_xy) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    keep_points = np.ones(len(segmentation_points_xy), dtype=bool)

    for index in range(len(segmentation_points_xy)):
        for compare_index in range(index + 1, len(segmentation_points_xy)):
            distance = np.linalg.norm(segmentation_points_xy[index] - segmentation_points_xy[compare_index])
            if distance < distance_threshold:
                keep_points[compare_index] = False

    return segmentation_points_xy[keep_points].astype(np.int32)


def _average_path_radius(
    shortest_path_rc: np.ndarray,
    shortest_path_length: int,
    point_data: dict[tuple[int, int], float],
) -> float:
    if shortest_path_length <= 0:
        return float("nan")

    average_radius = 0.0
    for point in shortest_path_rc[:shortest_path_length]:
        average_radius += get_radius(point_data, point)

    return average_radius / float(shortest_path_length)


def _detect_stenosis_from_queue(
    queue_rc: np.ndarray,
    average_radius: float,
    point_data: dict[tuple[int, int], float],
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if len(queue_rc) < 3:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float64)

    middle_points: list[tuple[int, int]] = []
    stenosis_degrees: list[float] = []

    for index in range(1, len(queue_rc) - 1, 3):
        previous_radius = get_radius(point_data, queue_rc[index - 1])
        current_radius = get_radius(point_data, queue_rc[index])
        next_radius = get_radius(point_data, queue_rc[index + 1])
        denominator = previous_radius + next_radius

        if denominator == 0:
            stenosis_ratio = float("inf")
        else:
            stenosis_ratio = 2.0 * current_radius / denominator

        stenosis_degree = 1.0 - stenosis_ratio

        if stenosis_degree > config.stenosis_threshold and average_radius > config.average_radius_threshold:
            middle_points.append((int(queue_rc[index, 0]), int(queue_rc[index, 1])))
            stenosis_degrees.append(float(stenosis_degree))

    if not middle_points:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float64)

    return np.asarray(middle_points, dtype=np.int32), np.asarray(stenosis_degrees, dtype=np.float64)


def _finalize_stenosis_points(
    raw_stenosis_points_rc: list[tuple[int, int]],
    raw_stenosis_degrees: list[float],
    final_point_distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not raw_stenosis_points_rc:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float64)

    stenosis_points_rc = np.asarray(raw_stenosis_points_rc, dtype=np.int32)
    stenosis_points_xy = np.fliplr(stenosis_points_rc).astype(np.int32)

    # Preserve the MATLAB behavior exactly: sort the points but do not
    # permute the degree array before applying the final keep mask.
    sort_order = np.lexsort((stenosis_points_xy[:, 1], stenosis_points_xy[:, 0]))
    stenosis_points_xy = stenosis_points_xy[sort_order]

    keep_indices = np.ones(len(stenosis_points_xy), dtype=bool)
    for index in range(len(stenosis_points_xy) - 1):
        if abs(int(stenosis_points_xy[index, 0]) - int(stenosis_points_xy[index + 1, 0])) < final_point_distance_threshold:
            if stenosis_points_xy[index, 0] < stenosis_points_xy[index + 1, 0]:
                keep_indices[index + 1] = False
            else:
                keep_indices[index] = False

    final_points_xy = stenosis_points_xy[keep_indices].astype(np.int32)
    final_degrees = np.asarray(raw_stenosis_degrees, dtype=np.float64)[keep_indices]
    return final_points_xy, final_degrees


def _classify_stenosis(stenosis_degree: float) -> str:
    if stenosis_degree > 0.75:
        return "severe"
    if stenosis_degree > 0.5:
        return "moderate"
    return "mild"


def _serialize_points_xy(points_xy: np.ndarray) -> list[list[int]]:
    if points_xy.size == 0:
        return []

    return [[int(point[0]), int(point[1])] for point in points_xy]


def _extract_frame_index(image_path: Path) -> int | None:
    match = SLICE_FRAME_PATTERN.match(image_path.stem)
    if match is None:
        return None

    return int(match.group(1))


def _build_view_id(image_path: Path) -> str:
    return image_path.parent.as_posix()
