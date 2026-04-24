from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class LesionObservation:
    frame_index: int | None
    image_name: str
    point_xy: np.ndarray
    degree: float
    severity: str
    registered_point_xy: np.ndarray | None = None
    reference_centerline_point_xy: np.ndarray | None = None
    centerline_distance: float | None = None
    centerline_component_id: int | None = None
    centerline_arc_length: float | None = None
    centerline_position: float | None = None
    centerline_component_length: float | None = None
    centerline_position_status: str | None = None
    centerline_position_reason: str | None = None

    @property
    def anchor_point_xy(self) -> np.ndarray:
        if self.reference_centerline_point_xy is not None:
            return self.reference_centerline_point_xy.astype(np.int32)
        if self.registered_point_xy is not None:
            return np.rint(self.registered_point_xy).astype(np.int32)
        return self.point_xy.astype(np.int32)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_index": self.frame_index,
            "image_name": self.image_name,
            "x": int(self.point_xy[0]),
            "y": int(self.point_xy[1]),
            "degree": float(self.degree),
            "severity": self.severity,
        }

        registered_point = _float_point_to_dict(self.registered_point_xy)
        if registered_point is not None:
            payload["registered_point"] = registered_point

        reference_point = _int_point_to_dict(self.reference_centerline_point_xy)
        if reference_point is not None:
            payload["reference_centerline_point"] = reference_point

        if self.centerline_distance is not None:
            payload["centerline_distance"] = float(self.centerline_distance)
        if self.centerline_component_id is not None:
            payload["centerline_component_id"] = int(self.centerline_component_id)
        if self.centerline_arc_length is not None:
            payload["centerline_arc_length"] = float(self.centerline_arc_length)
        if self.centerline_position is not None:
            payload["centerline_position"] = float(self.centerline_position)
        if self.centerline_component_length is not None:
            payload["centerline_component_length"] = float(self.centerline_component_length)
        if self.centerline_position_status is not None:
            payload["centerline_position_status"] = self.centerline_position_status
        if self.centerline_position_reason is not None:
            payload["centerline_position_reason"] = self.centerline_position_reason

        return payload


@dataclass(slots=True)
class FrameLevelResult:
    result_path: Path
    image_path: str | None
    mask_path: str | None
    image_name: str
    image_stem: str
    view_id: str
    frame_index: int | None
    width: int | None
    height: int | None
    skeleton_points_xy: np.ndarray
    observations: list[LesionObservation]

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def frame_id(self) -> str:
        return self.image_stem

    @property
    def stenosis_points_xy(self) -> np.ndarray:
        if not self.observations:
            return np.zeros((0, 2), dtype=np.int32)

        return np.asarray([observation.point_xy for observation in self.observations], dtype=np.int32)

    @property
    def stenosis_degrees(self) -> np.ndarray:
        if not self.observations:
            return np.zeros((0,), dtype=np.float64)

        return np.asarray([observation.degree for observation in self.observations], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "result_path": str(self.result_path),
            "frame_id": self.frame_id,
            "image_name": self.image_name,
            "image_stem": self.image_stem,
            "view_id": self.view_id,
            "frame_index": self.frame_index,
            "observation_count": self.observation_count,
            "skeleton_point_count": int(len(self.skeleton_points_xy)),
        }

        if self.image_path is not None:
            payload["image_path"] = self.image_path
        if self.mask_path is not None:
            payload["mask_path"] = self.mask_path
        if self.width is not None:
            payload["width"] = int(self.width)
        if self.height is not None:
            payload["height"] = int(self.height)

        return payload


@dataclass(slots=True)
class ViewSequence:
    view_id: str
    frames: list[FrameLevelResult]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "frame_count": self.frame_count,
            "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass(slots=True)
class ReferenceFrameSelection:
    frame: FrameLevelResult
    method: str
    score_name: str
    score_value: float
    support_point_count: int
    support_coverage_ratio: float
    middle_frame_distance: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.to_dict(),
            "method": self.method,
            "score_name": self.score_name,
            "score_value": float(self.score_value),
            "support_point_count": int(self.support_point_count),
            "support_coverage_ratio": float(self.support_coverage_ratio),
            "middle_frame_distance": int(self.middle_frame_distance),
            "reason": self.reason,
        }


@dataclass(slots=True)
class FrameRegistration:
    image_name: str
    frame_index: int | None
    reference_image_name: str
    reference_frame_index: int | None
    transform_matrix: np.ndarray
    method: str
    status: str
    score: float | None = None
    fallback_used: bool = False
    failure_reason: str | None = None

    @property
    def translation_xy(self) -> np.ndarray:
        return self.transform_matrix[:, 2].astype(np.float64)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_name": self.image_name,
            "frame_index": self.frame_index,
            "reference_image_name": self.reference_image_name,
            "reference_frame_index": self.reference_frame_index,
            "translation_xy": _float_point_to_dict(self.translation_xy),
            "transform_matrix": [[float(value) for value in row] for row in self.transform_matrix],
            "method": self.method,
            "status": self.status,
            "fallback_used": self.fallback_used,
        }

        if self.score is not None:
            payload["score"] = float(self.score)
        if self.failure_reason is not None:
            payload["failure_reason"] = self.failure_reason

        return payload


@dataclass(slots=True)
class LesionTrack:
    track_id: int
    observations: list[LesionObservation] = field(default_factory=list)

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def supporting_frame_count(self) -> int:
        return len(self.ordered_observations)

    @property
    def frame_indices(self) -> list[int]:
        return sorted({observation.frame_index for observation in self.observations if observation.frame_index is not None})

    @property
    def ordered_observations(self) -> list[LesionObservation]:
        return sorted(self.observations, key=_observation_sort_key)

    @property
    def supporting_frames(self) -> list[dict[str, int | str | None]]:
        return [
            {
                "frame_index": observation.frame_index,
                "image_name": observation.image_name,
            }
            for observation in self.ordered_observations
        ]

    @property
    def last_observation(self) -> LesionObservation | None:
        if not self.observations:
            return None
        return self.ordered_observations[-1]

    @property
    def frame_span(self) -> int:
        if not self.observations:
            return 0

        ordered_observations = self.ordered_observations
        frame_indices = [observation.frame_index for observation in ordered_observations if observation.frame_index is not None]
        if len(frame_indices) == len(ordered_observations):
            return int(frame_indices[-1] - frame_indices[0] + 1)

        return len(ordered_observations)

    @property
    def coverage_ratio(self) -> float:
        if self.frame_span <= 0:
            return 0.0
        return float(self.supporting_frame_count / self.frame_span)

    @property
    def max_frame_gap(self) -> int:
        ordered_observations = self.ordered_observations
        if len(ordered_observations) < 2:
            return 0

        frame_indices = [observation.frame_index for observation in ordered_observations if observation.frame_index is not None]
        if len(frame_indices) != len(ordered_observations):
            return 0

        frame_gaps = [frame_indices[index] - frame_indices[index - 1] for index in range(1, len(frame_indices))]
        return int(max(frame_gaps, default=0))

    @property
    def registered_points_xy(self) -> np.ndarray:
        if not self.observations:
            return np.zeros((0, 2), dtype=np.float64)

        return np.asarray(
            [
                observation.registered_point_xy if observation.registered_point_xy is not None else observation.point_xy.astype(np.float64)
                for observation in self.ordered_observations
            ],
            dtype=np.float64,
        )

    @property
    def representative_registered_point_xy(self) -> np.ndarray | None:
        if not self.observations:
            return None

        return self.registered_points_xy.mean(axis=0)

    @property
    def representative_point_xy(self) -> np.ndarray | None:
        if not self.observations:
            return None

        anchor_points = np.asarray([observation.anchor_point_xy for observation in self.observations], dtype=np.float64)
        return np.rint(anchor_points.mean(axis=0)).astype(np.int32)

    @property
    def representative_centerline_point_xy(self) -> np.ndarray | None:
        centerline_points = [
            observation.reference_centerline_point_xy.astype(np.float64)
            for observation in self.ordered_observations
            if observation.reference_centerline_point_xy is not None
        ]
        if not centerline_points:
            return None

        return np.rint(np.asarray(centerline_points, dtype=np.float64).mean(axis=0)).astype(np.int32)

    @property
    def centerline_positions(self) -> np.ndarray:
        values = [
            float(observation.centerline_position)
            for observation in self.ordered_observations
            if observation.centerline_position is not None
        ]
        if not values:
            return np.zeros((0,), dtype=np.float64)

        return np.asarray(values, dtype=np.float64)

    @property
    def dominant_centerline_component_id(self) -> int | None:
        component_ids = [
            int(observation.centerline_component_id)
            for observation in self.ordered_observations
            if observation.centerline_component_id is not None
        ]
        if not component_ids:
            return None

        counts: dict[int, int] = {}
        for component_id in component_ids:
            counts[component_id] = counts.get(component_id, 0) + 1

        return min(
            counts,
            key=lambda component_id: (-counts[component_id], component_id),
        )

    @property
    def mean_centerline_position(self) -> float | None:
        if self.centerline_positions.size == 0:
            return None
        return float(np.mean(self.centerline_positions, dtype=np.float64))

    @property
    def centerline_position_std(self) -> float | None:
        if self.centerline_positions.size == 0:
            return None
        return float(np.std(self.centerline_positions, dtype=np.float64))

    @property
    def degree_values(self) -> np.ndarray:
        if not self.observations:
            return np.zeros((0,), dtype=np.float64)

        return np.asarray([observation.degree for observation in self.ordered_observations], dtype=np.float64)

    @property
    def max_degree(self) -> float:
        if self.degree_values.size == 0:
            return 0.0
        return float(np.max(self.degree_values))

    @property
    def mean_degree(self) -> float:
        if self.degree_values.size == 0:
            return 0.0
        return float(np.mean(self.degree_values, dtype=np.float64))

    @property
    def min_degree(self) -> float:
        if self.degree_values.size == 0:
            return 0.0
        return float(np.min(self.degree_values))

    @property
    def degree_std(self) -> float:
        if self.degree_values.size == 0:
            return 0.0
        return float(np.std(self.degree_values, dtype=np.float64))

    @property
    def image_position_std_px(self) -> float | None:
        representative_point_xy = self.representative_registered_point_xy
        if representative_point_xy is None:
            return None

        deltas = self.registered_points_xy - representative_point_xy
        distances = np.linalg.norm(deltas, axis=1)
        return float(np.std(distances, dtype=np.float64))

    @property
    def severity(self) -> str:
        return classify_stenosis(self.max_degree)

    def to_dict(self) -> dict[str, Any]:
        positions_payload: dict[str, Any] = {
            "representative_registered_point": _float_point_to_dict(self.representative_registered_point_xy),
            "representative_centerline_point": _int_point_to_dict(self.representative_centerline_point_xy),
            "registered_points": [_float_point_to_dict(point_xy) for point_xy in self.registered_points_xy],
            "centerline_positions": [float(value) for value in self.centerline_positions],
        }

        dominant_component_id = self.dominant_centerline_component_id
        if dominant_component_id is not None:
            positions_payload["dominant_centerline_component_id"] = int(dominant_component_id)
        if self.mean_centerline_position is not None:
            positions_payload["mean_centerline_position"] = float(self.mean_centerline_position)

        consistency_payload: dict[str, Any] = {
            "frame_span": int(self.frame_span),
            "coverage_ratio": float(self.coverage_ratio),
            "max_frame_gap": int(self.max_frame_gap),
            "degree_std": float(self.degree_std),
        }

        if self.image_position_std_px is not None:
            consistency_payload["image_position_std_px"] = float(self.image_position_std_px)
        if self.centerline_position_std is not None:
            consistency_payload["centerline_position_std"] = float(self.centerline_position_std)

        return {
            "track_id": self.track_id,
            "observation_count": self.supporting_frame_count,
            "frame_indices": self.frame_indices,
            "supporting_frames": self.supporting_frames,
            "representative_point": _int_point_to_dict(self.representative_point_xy),
            "max_degree": float(self.max_degree),
            "mean_degree": float(self.mean_degree),
            "positions": positions_payload,
            "degrees": {
                "values": [float(value) for value in self.degree_values],
                "min": float(self.min_degree),
                "max": float(self.max_degree),
                "mean": float(self.mean_degree),
                "std": float(self.degree_std),
            },
            "consistency": consistency_payload,
            "severity": self.severity,
            "observations": [observation.to_dict() for observation in self.ordered_observations],
        }


@dataclass(slots=True)
class PersistentLesion:
    lesion_id: int
    track_id: int
    supporting_frames: list[dict[str, int | str | None]]
    frame_indices: list[int]
    supporting_frame_count: int
    total_frame_count: int
    persistence_ratio: float
    median_degree: float
    max_degree: float
    median_registered_point_xy: np.ndarray | None
    median_centerline_point_xy: np.ndarray | None
    median_centerline_position: float | None
    dominant_centerline_component_id: int | None
    degree_std: float
    image_position_std_px: float | None
    centerline_position_std: float | None
    max_frame_gap: int

    @property
    def severity(self) -> str:
        return classify_stenosis(self.median_degree)

    def to_dict(self) -> dict[str, Any]:
        positions_payload: dict[str, Any] = {
            "median_registered_point": _float_point_to_dict(self.median_registered_point_xy),
            "median_centerline_point": _int_point_to_dict(self.median_centerline_point_xy),
        }

        if self.median_centerline_position is not None:
            positions_payload["median_centerline_position"] = float(self.median_centerline_position)
        if self.dominant_centerline_component_id is not None:
            positions_payload["dominant_centerline_component_id"] = int(self.dominant_centerline_component_id)

        stability_payload: dict[str, Any] = {
            "degree_std": float(self.degree_std),
            "max_frame_gap": int(self.max_frame_gap),
        }

        if self.image_position_std_px is not None:
            stability_payload["image_position_std_px"] = float(self.image_position_std_px)
        if self.centerline_position_std is not None:
            stability_payload["centerline_position_std"] = float(self.centerline_position_std)

        return {
            "lesion_id": self.lesion_id,
            "track_id": self.track_id,
            "severity": self.severity,
            "supporting_frame_count": int(self.supporting_frame_count),
            "total_frame_count": int(self.total_frame_count),
            "persistence_ratio": float(self.persistence_ratio),
            "frame_indices": self.frame_indices,
            "supporting_frames": self.supporting_frames,
            "positions": positions_payload,
            "degrees": {
                "median": float(self.median_degree),
                "max": float(self.max_degree),
            },
            "stability": stability_payload,
        }


@dataclass(slots=True)
class ViewLevelResult:
    view_id: str
    frames: list[FrameLevelResult]
    reference_selection: ReferenceFrameSelection
    registrations: list[FrameRegistration]
    tracks: list[LesionTrack]
    persistent_lesions: list[PersistentLesion]
    final_lesion: PersistentLesion | None
    min_supporting_frames: int
    min_persistence_ratio: float
    final_selection_rule: str

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def reference_frame(self) -> FrameLevelResult:
        return self.reference_selection.frame

    @property
    def persistent_lesion_count(self) -> int:
        return len(self.persistent_lesions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "frame_count": self.frame_count,
            "frames": [frame.to_dict() for frame in self.frames],
            "reference_selection": self.reference_selection.to_dict(),
            "registrations": [registration.to_dict() for registration in self.registrations],
            "fusion": {
                "track_count": len(self.tracks),
                "persistent_lesion_count": self.persistent_lesion_count,
                "min_supporting_frames": int(self.min_supporting_frames),
                "min_persistence_ratio": float(self.min_persistence_ratio),
                "final_selection_rule": self.final_selection_rule,
            },
            "final_lesion": self.final_lesion.to_dict() if self.final_lesion is not None else None,
            "persistent_lesions": [persistent_lesion.to_dict() for persistent_lesion in self.persistent_lesions],
            "tracks": [track.to_dict() for track in self.tracks],
        }


def classify_stenosis(stenosis_degree: float) -> str:
    if stenosis_degree > 0.75:
        return "severe"
    if stenosis_degree > 0.5:
        return "moderate"
    return "mild"


def _int_point_to_dict(point_xy: np.ndarray | None) -> dict[str, int] | None:
    if point_xy is None:
        return None

    return {
        "x": int(point_xy[0]),
        "y": int(point_xy[1]),
    }


def _float_point_to_dict(point_xy: np.ndarray | None) -> dict[str, float] | None:
    if point_xy is None:
        return None

    return {
        "x": float(point_xy[0]),
        "y": float(point_xy[1]),
    }


def _observation_sort_key(observation: LesionObservation) -> tuple[int, int | str, str]:
    if observation.frame_index is not None:
        return (0, observation.frame_index, observation.image_name)

    return (1, observation.image_name, observation.image_name)
