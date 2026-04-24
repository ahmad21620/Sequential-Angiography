from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import LesionObservation, LesionTrack


DEFAULT_MAX_REGISTERED_DISTANCE = 18.0
DEFAULT_MAX_CENTERLINE_POSITION_DELTA = 0.08
DEFAULT_MAX_DEGREE_DELTA = 0.2
DEFAULT_MAX_FRAME_GAP = 2
DEFAULT_FRAME_GAP_PENALTY = 0.05


@dataclass(frozen=True, slots=True)
class _TrackMatchCandidate:
    track_index: int
    observation_index: int
    score: float
    registered_distance: float
    centerline_position_delta: float | None
    degree_delta: float
    frame_gap: int | None


def build_lesion_tracks(
    observations: list[LesionObservation],
    *,
    max_registered_distance: float = DEFAULT_MAX_REGISTERED_DISTANCE,
    max_centerline_position_delta: float = DEFAULT_MAX_CENTERLINE_POSITION_DELTA,
    max_degree_delta: float = DEFAULT_MAX_DEGREE_DELTA,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
) -> list[LesionTrack]:
    if max_registered_distance <= 0.0:
        raise ValueError("max_registered_distance must be positive.")
    if max_centerline_position_delta <= 0.0:
        raise ValueError("max_centerline_position_delta must be positive.")
    if max_degree_delta <= 0.0:
        raise ValueError("max_degree_delta must be positive.")
    if max_frame_gap < 1:
        raise ValueError("max_frame_gap must be at least 1.")

    tracks: list[LesionTrack] = []

    for frame_observations in _group_observations_by_frame(observations):
        match_candidates = _collect_match_candidates(
            tracks,
            frame_observations,
            max_registered_distance=max_registered_distance,
            max_centerline_position_delta=max_centerline_position_delta,
            max_degree_delta=max_degree_delta,
            max_frame_gap=max_frame_gap,
        )
        matched_observation_indices = _assign_frame_matches(tracks, frame_observations, match_candidates)

        for observation_index, observation in enumerate(frame_observations):
            if observation_index in matched_observation_indices:
                continue

            tracks.append(LesionTrack(track_id=len(tracks) + 1, observations=[observation]))

    return tracks


def _group_observations_by_frame(observations: list[LesionObservation]) -> list[list[LesionObservation]]:
    ordered_observations = sorted(observations, key=_observation_sort_key)
    frame_groups: list[list[LesionObservation]] = []

    for observation in ordered_observations:
        frame_key = _observation_frame_key(observation)
        if not frame_groups or _observation_frame_key(frame_groups[-1][0]) != frame_key:
            frame_groups.append([observation])
            continue

        frame_groups[-1].append(observation)

    return frame_groups


def _collect_match_candidates(
    tracks: list[LesionTrack],
    frame_observations: list[LesionObservation],
    *,
    max_registered_distance: float,
    max_centerline_position_delta: float,
    max_degree_delta: float,
    max_frame_gap: int,
) -> list[_TrackMatchCandidate]:
    candidates: list[_TrackMatchCandidate] = []

    for observation_index, observation in enumerate(frame_observations):
        for track_index, track in enumerate(tracks):
            candidate = _evaluate_track_match(
                track,
                observation,
                track_index=track_index,
                observation_index=observation_index,
                max_registered_distance=max_registered_distance,
                max_centerline_position_delta=max_centerline_position_delta,
                max_degree_delta=max_degree_delta,
                max_frame_gap=max_frame_gap,
            )
            if candidate is not None:
                candidates.append(candidate)

    return candidates


def _assign_frame_matches(
    tracks: list[LesionTrack],
    frame_observations: list[LesionObservation],
    candidates: list[_TrackMatchCandidate],
) -> set[int]:
    assigned_track_indices: set[int] = set()
    assigned_observation_indices: set[int] = set()

    for candidate in sorted(candidates, key=_candidate_sort_key):
        if candidate.track_index in assigned_track_indices:
            continue
        if candidate.observation_index in assigned_observation_indices:
            continue

        tracks[candidate.track_index].observations.append(frame_observations[candidate.observation_index])
        assigned_track_indices.add(candidate.track_index)
        assigned_observation_indices.add(candidate.observation_index)

    return assigned_observation_indices


def _evaluate_track_match(
    track: LesionTrack,
    observation: LesionObservation,
    *,
    track_index: int,
    observation_index: int,
    max_registered_distance: float,
    max_centerline_position_delta: float,
    max_degree_delta: float,
    max_frame_gap: int,
) -> _TrackMatchCandidate | None:
    last_observation = track.last_observation
    if last_observation is None:
        return None

    frame_gap = _frame_gap(last_observation, observation)
    if frame_gap is not None:
        if frame_gap <= 0 or frame_gap > max_frame_gap:
            return None
    elif _observation_frame_key(last_observation) == _observation_frame_key(observation):
        return None

    track_component_id = track.dominant_centerline_component_id
    observation_component_id = observation.centerline_component_id
    if (
        track_component_id is not None
        and observation_component_id is not None
        and track_component_id != observation_component_id
    ):
        return None

    representative_registered_point_xy = track.representative_registered_point_xy
    if representative_registered_point_xy is None or observation.registered_point_xy is None:
        return None

    registered_distance = float(
        np.linalg.norm(observation.registered_point_xy.astype(np.float64) - representative_registered_point_xy.astype(np.float64))
    )
    if registered_distance > max_registered_distance:
        return None

    degree_delta = abs(float(observation.degree) - float(track.mean_degree))
    if degree_delta > max_degree_delta:
        return None

    centerline_position_delta: float | None = None
    track_centerline_position = track.mean_centerline_position
    if track_centerline_position is not None and observation.centerline_position is not None:
        centerline_position_delta = abs(float(observation.centerline_position) - float(track_centerline_position))
        if centerline_position_delta > max_centerline_position_delta:
            return None

    score = _compute_match_score(
        registered_distance=registered_distance,
        centerline_position_delta=centerline_position_delta,
        degree_delta=degree_delta,
        frame_gap=frame_gap,
        max_registered_distance=max_registered_distance,
        max_centerline_position_delta=max_centerline_position_delta,
        max_degree_delta=max_degree_delta,
    )

    return _TrackMatchCandidate(
        track_index=track_index,
        observation_index=observation_index,
        score=score,
        registered_distance=registered_distance,
        centerline_position_delta=centerline_position_delta,
        degree_delta=degree_delta,
        frame_gap=frame_gap,
    )


def _compute_match_score(
    *,
    registered_distance: float,
    centerline_position_delta: float | None,
    degree_delta: float,
    frame_gap: int | None,
    max_registered_distance: float,
    max_centerline_position_delta: float,
    max_degree_delta: float,
) -> float:
    score_terms = [
        registered_distance / max_registered_distance,
        degree_delta / max_degree_delta,
    ]

    if centerline_position_delta is not None:
        score_terms.append(centerline_position_delta / max_centerline_position_delta)

    score = float(np.mean(score_terms, dtype=np.float64))
    if frame_gap is not None and frame_gap > 1:
        score += DEFAULT_FRAME_GAP_PENALTY * float(frame_gap - 1)

    return score


def _frame_gap(previous_observation: LesionObservation, observation: LesionObservation) -> int | None:
    if previous_observation.frame_index is None or observation.frame_index is None:
        return None

    return int(observation.frame_index - previous_observation.frame_index)


def _candidate_sort_key(candidate: _TrackMatchCandidate) -> tuple[float, float, float, float, float, int, int]:
    centerline_delta = candidate.centerline_position_delta
    if centerline_delta is None:
        centerline_delta = float("inf")

    frame_gap = candidate.frame_gap
    if frame_gap is None:
        frame_gap = 0

    return (
        float(candidate.score),
        float(candidate.registered_distance),
        float(centerline_delta),
        float(candidate.degree_delta),
        float(frame_gap),
        int(candidate.track_index),
        int(candidate.observation_index),
    )


def _observation_sort_key(observation: LesionObservation) -> tuple[int, int | str, float, str]:
    if observation.frame_index is not None:
        return (0, observation.frame_index, -float(observation.degree), observation.image_name)

    return (1, observation.image_name, -float(observation.degree), observation.image_name)


def _observation_frame_key(observation: LesionObservation) -> tuple[int | None, str]:
    return observation.frame_index, observation.image_name
