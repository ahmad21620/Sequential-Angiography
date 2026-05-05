from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .loader import load_frame_results, sort_frame_results
from .mapping import map_observations_to_reference_centerline
from .models import (
    FrameLevelResult,
    FrameRegistration,
    LesionTrack,
    PersistentLesion,
    ReferenceFrameSelection,
    ViewLevelResult,
    ViewSequence,
)
from .reference import select_reference_frame
from .registration import build_frame_registrations
from .tracking import build_lesion_tracks


DEFAULT_MIN_SUPPORTING_FRAMES = 3
DEFAULT_MIN_PERSISTENCE_RATIO = 0.25
FINAL_LESION_SELECTION_RULE = (
    "highest_median_degree_then_highest_max_degree_then_highest_persistence_ratio_then_lowest_track_id"
)


@dataclass(frozen=True, slots=True)
class TemporalFusionConfig:
    min_supporting_frames: int = DEFAULT_MIN_SUPPORTING_FRAMES
    min_persistence_ratio: float = DEFAULT_MIN_PERSISTENCE_RATIO


@dataclass(frozen=True, slots=True)
class _TemporalFusionBase:
    view_id: str
    frame_results: list[FrameLevelResult]
    reference_selection: ReferenceFrameSelection
    registrations: list[FrameRegistration]
    tracks: list[LesionTrack]


def run_temporal_fusion(
    frame_result_paths: list[str | Path],
    *,
    min_supporting_frames: int = DEFAULT_MIN_SUPPORTING_FRAMES,
    min_persistence_ratio: float = DEFAULT_MIN_PERSISTENCE_RATIO,
) -> ViewLevelResult:
    frame_results = sort_frame_results(load_frame_results(frame_result_paths))
    if not frame_results:
        raise ValueError("At least one frame-level stenosis result is required for temporal fusion.")
    view_sequence = ViewSequence(
        view_id=_resolve_view_id(frame_results),
        frames=frame_results,
    )

    return run_temporal_fusion_on_view_sequence(
        view_sequence,
        min_supporting_frames=min_supporting_frames,
        min_persistence_ratio=min_persistence_ratio,
    )


def run_temporal_fusion_on_view_sequence(
    view_sequence: ViewSequence,
    *,
    min_supporting_frames: int = DEFAULT_MIN_SUPPORTING_FRAMES,
    min_persistence_ratio: float = DEFAULT_MIN_PERSISTENCE_RATIO,
) -> ViewLevelResult:
    return run_temporal_fusion_variants_on_view_sequence(
        view_sequence,
        [
            TemporalFusionConfig(
                min_supporting_frames=min_supporting_frames,
                min_persistence_ratio=min_persistence_ratio,
            )
        ],
    )[0]


def run_temporal_fusion_variants_on_view_sequence(
    view_sequence: ViewSequence,
    configs: list[TemporalFusionConfig],
) -> list[ViewLevelResult]:
    if not configs:
        raise ValueError("At least one temporal fusion config is required.")
    for config in configs:
        _validate_persistence_thresholds(
            min_supporting_frames=config.min_supporting_frames,
            min_persistence_ratio=config.min_persistence_ratio,
        )

    fusion_base = _build_temporal_fusion_base(view_sequence)
    results: list[ViewLevelResult] = []
    for config in configs:
        persistent_lesions = build_persistent_lesions(
            fusion_base.tracks,
            total_frame_count=len(fusion_base.frame_results),
            min_supporting_frames=config.min_supporting_frames,
            min_persistence_ratio=config.min_persistence_ratio,
        )
        final_lesion = select_final_view_lesion(persistent_lesions)
        results.append(
            ViewLevelResult(
                view_id=fusion_base.view_id,
                frames=fusion_base.frame_results,
                reference_selection=fusion_base.reference_selection,
                registrations=fusion_base.registrations,
                tracks=fusion_base.tracks,
                persistent_lesions=persistent_lesions,
                final_lesion=final_lesion,
                min_supporting_frames=config.min_supporting_frames,
                min_persistence_ratio=config.min_persistence_ratio,
                final_selection_rule=FINAL_LESION_SELECTION_RULE,
            )
        )
    return results


def _build_temporal_fusion_base(view_sequence: ViewSequence) -> _TemporalFusionBase:
    frame_results = sort_frame_results(list(view_sequence.frames))
    if not frame_results:
        raise ValueError("At least one frame-level stenosis result is required for temporal fusion.")

    view_id = view_sequence.view_id or _resolve_view_id(frame_results)
    reference_selection = select_reference_frame(frame_results)
    reference_frame = reference_selection.frame
    registrations = build_frame_registrations(frame_results, reference_frame)
    registration_by_image_name = {registration.image_name: registration for registration in registrations}

    mapped_observations = []
    for frame_result in frame_results:
        mapped_observations.extend(
            map_observations_to_reference_centerline(
                frame_result,
                reference_frame,
                registration_by_image_name[frame_result.image_name],
            )
        )

    tracks = build_lesion_tracks(mapped_observations)
    return _TemporalFusionBase(
        view_id=view_id,
        frame_results=frame_results,
        reference_selection=reference_selection,
        registrations=registrations,
        tracks=tracks,
    )


def save_view_level_result(view_result: ViewLevelResult, output_path: str | Path) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(view_result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return resolved_output_path


def _resolve_view_id(frame_results: list[FrameLevelResult]) -> str:
    view_ids = {frame_result.view_id for frame_result in frame_results if frame_result.view_id}
    if not view_ids:
        return ""
    if len(view_ids) > 1:
        raise ValueError(f"Temporal fusion requires frames from one view, got multiple view ids: {sorted(view_ids)}")
    return next(iter(view_ids))


def build_persistent_lesions(
    tracks: list[LesionTrack],
    *,
    total_frame_count: int,
    min_supporting_frames: int = DEFAULT_MIN_SUPPORTING_FRAMES,
    min_persistence_ratio: float = DEFAULT_MIN_PERSISTENCE_RATIO,
) -> list[PersistentLesion]:
    if total_frame_count < 1:
        raise ValueError("total_frame_count must be at least 1.")
    _validate_persistence_thresholds(
        min_supporting_frames=min_supporting_frames,
        min_persistence_ratio=min_persistence_ratio,
    )

    persistent_lesions: list[PersistentLesion] = []

    for track in tracks:
        supporting_frames = track.supporting_frames
        supporting_frame_count = track.supporting_frame_count
        persistence_ratio = float(supporting_frame_count / total_frame_count)
        if supporting_frame_count < min_supporting_frames:
            continue
        if persistence_ratio < min_persistence_ratio:
            continue

        persistent_lesions.append(
            _summarize_persistent_lesion(
                track,
                supporting_frames=supporting_frames,
                lesion_id=0,
                total_frame_count=total_frame_count,
                persistence_ratio=persistence_ratio,
            )
        )

    persistent_lesions = sorted(persistent_lesions, key=_persistent_lesion_sort_key)
    for lesion_index, persistent_lesion in enumerate(persistent_lesions, start=1):
        persistent_lesion.lesion_id = lesion_index

    return persistent_lesions


def select_final_view_lesion(persistent_lesions: list[PersistentLesion]) -> PersistentLesion | None:
    if not persistent_lesions:
        return None

    return min(persistent_lesions, key=_persistent_lesion_sort_key)


def _summarize_persistent_lesion(
    track: LesionTrack,
    *,
    supporting_frames: list[dict[str, int | str | None]],
    lesion_id: int,
    total_frame_count: int,
    persistence_ratio: float,
) -> PersistentLesion:
    return PersistentLesion(
        lesion_id=lesion_id,
        track_id=track.track_id,
        supporting_frames=supporting_frames,
        frame_indices=track.frame_indices,
        supporting_frame_count=track.supporting_frame_count,
        total_frame_count=total_frame_count,
        persistence_ratio=persistence_ratio,
        median_degree=_compute_track_median_degree(track),
        max_degree=float(track.max_degree),
        median_registered_point_xy=_compute_median_float_point(track.registered_points_xy),
        median_centerline_point_xy=_compute_track_median_centerline_point(track),
        median_centerline_position=_compute_track_median_centerline_position(track),
        dominant_centerline_component_id=track.dominant_centerline_component_id,
        degree_std=float(track.degree_std),
        image_position_std_px=track.image_position_std_px,
        centerline_position_std=track.centerline_position_std,
        max_frame_gap=int(track.max_frame_gap),
    )


def _compute_track_median_degree(track: LesionTrack) -> float:
    if track.degree_values.size == 0:
        return 0.0

    return float(np.median(track.degree_values))


def _compute_track_median_centerline_position(track: LesionTrack) -> float | None:
    if track.centerline_positions.size == 0:
        return None

    return float(np.median(track.centerline_positions))


def _compute_track_median_centerline_point(track: LesionTrack) -> np.ndarray | None:
    centerline_points = [
        observation.reference_centerline_point_xy.astype(np.float64)
        for observation in track.ordered_observations
        if observation.reference_centerline_point_xy is not None
    ]
    if not centerline_points:
        return None

    return np.rint(np.median(np.asarray(centerline_points, dtype=np.float64), axis=0)).astype(np.int32)


def _compute_median_float_point(points_xy: np.ndarray) -> np.ndarray | None:
    if points_xy.size == 0:
        return None

    return np.median(points_xy.astype(np.float64), axis=0)


def _validate_persistence_thresholds(
    *,
    min_supporting_frames: int,
    min_persistence_ratio: float,
) -> None:
    if min_supporting_frames < 1:
        raise ValueError("min_supporting_frames must be at least 1.")
    if not 0.0 < min_persistence_ratio <= 1.0:
        raise ValueError("min_persistence_ratio must be in the range (0.0, 1.0].")


def _persistent_lesion_sort_key(persistent_lesion: PersistentLesion) -> tuple[float, float, float, int, int]:
    return (
        -float(persistent_lesion.median_degree),
        -float(persistent_lesion.max_degree),
        -float(persistent_lesion.persistence_ratio),
        -int(persistent_lesion.supporting_frame_count),
        int(persistent_lesion.track_id),
    )
