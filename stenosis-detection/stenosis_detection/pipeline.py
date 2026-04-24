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
    radius_search_range: float = 110.0
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
    mask_gray: np.ndarray
    binary_mask: np.ndarray
    skeleton_mask: np.ndarray
    skeleton_points_rc: np.ndarray
    skeleton_points_xy: np.ndarray
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
    mask_gray, binary_mask = _load_mask_image(resolved_mask_path, pipeline_config)

    skeleton_mask = thin_binary_mask(binary_mask)
    skeleton_points_rc = _skeleton_points_rc(skeleton_mask)
    skeleton_points_xy = _flip_points(skeleton_points_rc)

    point_data = build_point_data(skeleton_points_rc, mask_gray, pipeline_config.radius_search_range)

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
        mask_gray=mask_gray,
        binary_mask=binary_mask,
        skeleton_mask=skeleton_mask,
        skeleton_points_rc=skeleton_points_rc,
        skeleton_points_xy=skeleton_points_xy,
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
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Unable to read vessel mask: {mask_path}")

    mask = _ensure_uint8(mask)

    if mask.ndim == 3 and mask.shape[2] == 4:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR)

    resized_mask = _resize_image(mask, config.resize_width, config.resize_height)

    if resized_mask.ndim == 3:
        mask_gray = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = resized_mask

    if mask_gray.max(initial=0) <= 1:
        mask_gray = np.clip(mask_gray.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)

    binary_mask = mask_gray > 0
    return mask_gray, binary_mask


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
