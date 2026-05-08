from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


SCORE_SEMANTICS = "yolo_confidence_not_anatomical_stenosis_degree"
SEVERITY_SEMANTICS = "class_name_if_available_else_yolo_confidence_threshold_compatibility"
COORDINATE_SYSTEM = "MATLAB-compatible 1-based x/y pixel coordinates"

SLICE_FRAME_PATTERN = re.compile(r"^slice_(\d+)$", re.IGNORECASE)
CADICA_FRAME_PATTERN = re.compile(r"^p\d+_v\d+_(\d+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class YoloDetectorMetadata:
    name: str
    weights: str
    imgsz: int | None
    conf: float
    iou: float
    device: str | None
    score_semantics: str = SCORE_SEMANTICS
    severity_semantics: str = SEVERITY_SEMANTICS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class YoloDetection:
    bbox_xyxy_zero_based: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    image_width: int
    image_height: int

    @property
    def bbox_1based(self) -> dict[str, int]:
        x1_zero, y1_zero, x2_zero, y2_zero = self.bbox_xyxy_zero_based
        x1 = _clip_int(_round_half_up(x1_zero + 1.0), minimum=1, maximum=self.image_width)
        y1 = _clip_int(_round_half_up(y1_zero + 1.0), minimum=1, maximum=self.image_height)
        x2 = _clip_int(_round_half_up(x2_zero + 1.0), minimum=1, maximum=self.image_width)
        y2 = _clip_int(_round_half_up(y2_zero + 1.0), minimum=1, maximum=self.image_height)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": max(0, x2 - x1),
            "height": max(0, y2 - y1),
        }

    @property
    def center_xy_1based(self) -> tuple[int, int]:
        bbox = self.bbox_1based
        x = _clip_int(_round_half_up((bbox["x1"] + bbox["x2"]) / 2.0), minimum=1, maximum=self.image_width)
        y = _clip_int(_round_half_up((bbox["y1"] + bbox["y2"]) / 2.0), minimum=1, maximum=self.image_height)
        return x, y

    @property
    def severity(self) -> str:
        class_severity = _severity_from_class_name(self.class_name)
        if class_severity is not None:
            return class_severity
        return _severity_from_confidence(self.confidence)

    def to_stenosis_point_dict(self) -> dict[str, Any]:
        x, y = self.center_xy_1based
        confidence = float(self.confidence)
        return {
            "x": int(x),
            "y": int(y),
            "degree": confidence,
            "severity": self.severity,
            "confidence": confidence,
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "bbox": self.bbox_1based,
            "score_semantics": SCORE_SEMANTICS,
            "severity_semantics": SEVERITY_SEMANTICS,
        }

    def to_detection_dict(self) -> dict[str, Any]:
        center_x, center_y = self.center_xy_1based
        return {
            "x": int(center_x),
            "y": int(center_y),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "bbox": self.bbox_1based,
            "coordinate_system": COORDINATE_SYSTEM,
            "score_semantics": SCORE_SEMANTICS,
        }


def build_yolo_frame_payload(
    *,
    image_path: str | Path,
    mask_path: str | Path | None,
    detector: YoloDetectorMetadata,
    view_id: str,
    image_width: int,
    image_height: int,
    detections: list[YoloDetection],
    skeleton_points_xy: np.ndarray | list[list[int]] | None = None,
) -> dict[str, Any]:
    resolved_image_path = Path(image_path)
    skeleton_points = _serialize_points_xy(skeleton_points_xy)
    stenosis_points = [detection.to_stenosis_point_dict() for detection in detections]
    return {
        "image_path": str(resolved_image_path),
        "mask_path": None if mask_path is None else str(mask_path),
        "detector": detector.to_dict(),
        "frame": {
            "image_name": resolved_image_path.name,
            "image_stem": resolved_image_path.stem,
            "frame_index": extract_frame_index(resolved_image_path.stem),
            "view_id": view_id,
            "width": int(image_width),
            "height": int(image_height),
        },
        "coordinate_system": COORDINATE_SYSTEM,
        "counts": {
            "skeleton_points": len(skeleton_points),
            "stenosis_points": len(stenosis_points),
            "detections": len(detections),
        },
        "skeleton_points": skeleton_points,
        "stenosis_points": stenosis_points,
        "yolo_detections": [detection.to_detection_dict() for detection in detections],
        "metadata": {
            "degree_semantics": SCORE_SEMANTICS,
            "severity_semantics": SEVERITY_SEMANTICS,
            "bbox_coordinate_system": COORDINATE_SYSTEM,
            "note": (
                "YOLO confidence is copied into degree only for compatibility with "
                "temporal fusion, multi-view fusion, and benchmark code."
            ),
        },
    }


def extract_frame_index(image_stem: str) -> int | None:
    match = SLICE_FRAME_PATTERN.match(image_stem) or CADICA_FRAME_PATTERN.match(image_stem)
    if match is None:
        return None
    return int(match.group(1))


def _serialize_points_xy(points_xy: np.ndarray | list[list[int]] | None) -> list[list[int]]:
    if points_xy is None:
        return []
    array = np.asarray(points_xy, dtype=np.int32)
    if array.size == 0:
        return []
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("skeleton_points_xy must have shape (n, 2).")
    if np.any(array < 1):
        raise ValueError("skeleton_points_xy must use 1-based coordinates.")
    return [[int(point[0]), int(point[1])] for point in array]


def _severity_from_confidence(confidence: float) -> str:
    if confidence > 0.75:
        return "severe"
    if confidence > 0.50:
        return "moderate"
    return "mild"


def _severity_from_class_name(class_name: str) -> str | None:
    normalized = class_name.strip().lower()
    for severity in ("severe", "moderate", "mild"):
        if re.search(rf"\b{severity}\b", normalized):
            return severity
    return None


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _clip_int(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
