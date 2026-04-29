from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .models import FrameLevelResult, LesionObservation, ViewSequence, classify_stenosis


SLICE_FRAME_PATTERN = re.compile(r"^slice_(\d+)$", re.IGNORECASE)
FRAME_RESULT_SUFFIX = "_stenosis_results.json"
DEFAULT_VIEW_FRAME_COUNT = 12


class TemporalLoadError(ValueError):
    pass


def discover_frame_result_paths(result_source: str | Path) -> list[Path]:
    resolved_source = Path(result_source)
    if resolved_source.is_file():
        _validate_result_file_path(resolved_source)
        return [resolved_source]
    if not resolved_source.is_dir():
        raise FileNotFoundError(f"Frame result source does not exist: {resolved_source}")

    result_paths = sorted(path for path in resolved_source.rglob(f"*{FRAME_RESULT_SUFFIX}") if path.is_file())
    if not result_paths:
        raise TemporalLoadError(
            f"No frame-level stenosis result files matching '*{FRAME_RESULT_SUFFIX}' were found under: {resolved_source}"
        )

    return result_paths


def load_frame_result(result_path: str | Path) -> FrameLevelResult:
    resolved_path = Path(result_path)
    payload = _read_json_object(resolved_path)
    context = str(resolved_path)

    frame_payload = _require_object(payload, "frame", context=context)
    image_path = _coerce_optional_string(payload.get("image_path"))
    mask_path = _coerce_optional_string(payload.get("mask_path"))
    image_name = _require_non_empty_string(frame_payload, "image_name", context=context)
    image_stem = _require_non_empty_string(frame_payload, "image_stem", context=context)
    view_id = _require_non_empty_string(frame_payload, "view_id", context=context)
    width = _require_positive_int(frame_payload, "width", context=context)
    height = _require_positive_int(frame_payload, "height", context=context)

    frame_index = _coerce_optional_int(frame_payload.get("frame_index"))
    if frame_index is None:
        frame_index = _extract_frame_index(image_stem)

    if image_path is not None and image_name != Path(image_path).name:
        raise TemporalLoadError(
            f"{context}: frame.image_name='{image_name}' does not match the image_path filename '{Path(image_path).name}'."
        )

    skeleton_points_xy = _load_required_points_xy(payload, "skeleton_points", context=context)
    observations = [
        _load_observation(
            item,
            frame_index=frame_index,
            image_name=image_name,
            context=f"{context} -> stenosis_points[{index}]",
        )
        for index, item in enumerate(_require_array(payload, "stenosis_points", context=context))
    ]
    _validate_counts(payload, skeleton_points_xy, observations, context=context)

    return FrameLevelResult(
        result_path=resolved_path,
        image_path=image_path,
        mask_path=mask_path,
        image_name=image_name,
        image_stem=image_stem,
        view_id=view_id,
        frame_index=frame_index,
        width=width,
        height=height,
        skeleton_points_xy=skeleton_points_xy,
        observations=observations,
    )


def load_frame_results(result_source: str | Path | Sequence[str | Path]) -> list[FrameLevelResult]:
    result_paths = _normalize_result_paths(result_source)
    if not result_paths:
        raise TemporalLoadError("No frame-level stenosis result files were provided.")

    return [load_frame_result(result_path) for result_path in result_paths]


def load_view_sequences(
    result_source: str | Path | Sequence[str | Path],
    *,
    expected_frame_count: int | None = None,
) -> list[ViewSequence]:
    frame_results = load_frame_results(result_source)
    grouped_frames = _group_frame_results_by_view(frame_results)

    sequences: list[ViewSequence] = []
    for view_id in sorted(grouped_frames):
        ordered_frames = sort_frame_results(grouped_frames[view_id])
        _validate_view_sequence(view_id, ordered_frames, expected_frame_count=expected_frame_count)
        sequences.append(ViewSequence(view_id=view_id, frames=ordered_frames))

    return sequences


def load_view_sequence(
    result_source: str | Path | Sequence[str | Path],
    *,
    expected_frame_count: int | None = DEFAULT_VIEW_FRAME_COUNT,
    view_id: str | None = None,
) -> ViewSequence:
    if view_id is not None:
        sequences = load_view_sequences(result_source, expected_frame_count=None)
        sequence = _select_view_sequence(sequences, view_id)
        _validate_view_sequence(sequence.view_id, sequence.frames, expected_frame_count=expected_frame_count)
        return sequence

    sequences = load_view_sequences(result_source, expected_frame_count=expected_frame_count)

    if len(sequences) != 1:
        raise TemporalLoadError(
            f"Expected frame results from exactly one view, but found {len(sequences)} views: "
            f"{', '.join(sequence.view_id or '<empty>' for sequence in sequences)}"
        )

    return sequences[0]


def sort_frame_results(frame_results: list[FrameLevelResult]) -> list[FrameLevelResult]:
    missing_indices = [frame_result.image_name for frame_result in frame_results if frame_result.frame_index is None]
    if missing_indices:
        missing_preview = ", ".join(sorted(missing_indices))
        raise TemporalLoadError(
            f"Cannot sort frames without frame_index values. Missing frame_index for: {missing_preview}"
        )

    return sorted(frame_results, key=lambda frame_result: (int(frame_result.frame_index), frame_result.image_name))


def _select_view_sequence(sequences: list[ViewSequence], view_id: str) -> ViewSequence:
    matches = [sequence for sequence in sequences if sequence.view_id == view_id]
    if not matches:
        available_view_ids = ", ".join(sequence.view_id for sequence in sequences) or "<none>"
        raise TemporalLoadError(
            f"View '{view_id}' was not found in the provided frame results. Available view ids: {available_view_ids}"
        )

    return matches[0]


def _load_observation(
    payload: object,
    *,
    frame_index: int | None,
    image_name: str,
    context: str,
) -> LesionObservation:
    if not isinstance(payload, dict):
        raise TemporalLoadError(f"{context}: expected a JSON object, got {type(payload)!r}")

    degree = _require_float(payload, "degree", context=context)
    severity = _coerce_optional_string(payload.get("severity")) or classify_stenosis(degree)
    point_xy = np.asarray(
        [
            _require_positive_int(payload, "x", context=context),
            _require_positive_int(payload, "y", context=context),
        ],
        dtype=np.int32,
    )

    return LesionObservation(
        frame_index=frame_index,
        image_name=image_name,
        point_xy=point_xy,
        degree=degree,
        severity=severity,
    )


def _load_required_points_xy(payload: dict[str, Any], field_name: str, *, context: str) -> np.ndarray:
    raw_points = _require_array(payload, field_name, context=context)
    points_xy = np.asarray(raw_points, dtype=np.int32)
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise TemporalLoadError(f"{context}: field '{field_name}' must have shape (n, 2).")
    if np.any(points_xy < 1):
        raise TemporalLoadError(f"{context}: field '{field_name}' contains invalid 1-based coordinates.")

    return points_xy


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _extract_frame_index(image_stem: str) -> int | None:
    match = SLICE_FRAME_PATTERN.match(image_stem)
    if match is None:
        return None
    return int(match.group(1))


def _normalize_result_paths(result_source: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(result_source, (str, Path)):
        return discover_frame_result_paths(result_source)

    result_paths = [Path(item) for item in result_source]
    for result_path in result_paths:
        _validate_result_file_path(result_path)

    return result_paths


def _validate_result_file_path(result_path: Path) -> None:
    if not result_path.exists():
        raise FileNotFoundError(f"Frame result file does not exist: {result_path}")
    if not result_path.is_file():
        raise TemporalLoadError(f"Frame result path is not a file: {result_path}")
    if not result_path.name.endswith(FRAME_RESULT_SUFFIX):
        raise TemporalLoadError(
            f"Expected a frame result file ending with '{FRAME_RESULT_SUFFIX}', got: {result_path}"
        )


def _read_json_object(result_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TemporalLoadError(f"{result_path}: invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise TemporalLoadError(f"{result_path}: expected top-level JSON object, got {type(payload)!r}")

    return payload


def _require_object(payload: dict[str, Any], field_name: str, *, context: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise TemporalLoadError(
            f"{context}: missing required object field '{field_name}'. "
            f"Regenerate this frame result with the current pipeline."
        )

    return value


def _require_array(payload: dict[str, Any], field_name: str, *, context: str) -> list[Any]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise TemporalLoadError(
            f"{context}: missing required array field '{field_name}'. "
            f"Regenerate this frame result with the current pipeline."
        )

    return value


def _require_non_empty_string(payload: dict[str, Any], field_name: str, *, context: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise TemporalLoadError(f"{context}: missing required string field '{field_name}'.")

    return value


def _require_positive_int(payload: dict[str, Any], field_name: str, *, context: str) -> int:
    value = payload.get(field_name)
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise TemporalLoadError(f"{context}: field '{field_name}' must be an integer.") from exc

    if int_value < 1:
        raise TemporalLoadError(f"{context}: field '{field_name}' must be >= 1.")

    return int_value


def _require_float(payload: dict[str, Any], field_name: str, *, context: str) -> float:
    value = payload.get(field_name)
    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise TemporalLoadError(f"{context}: field '{field_name}' must be a float.") from exc

    if not np.isfinite(float_value):
        raise TemporalLoadError(f"{context}: field '{field_name}' must be finite.")

    return float_value


def _validate_counts(
    payload: dict[str, Any],
    skeleton_points_xy: np.ndarray,
    observations: list[LesionObservation],
    *,
    context: str,
) -> None:
    counts = payload.get("counts")
    if counts is None:
        return
    if not isinstance(counts, dict):
        raise TemporalLoadError(f"{context}: field 'counts' must be a JSON object.")

    expected_skeleton_count = counts.get("skeleton_points")
    if expected_skeleton_count is not None and int(expected_skeleton_count) != len(skeleton_points_xy):
        raise TemporalLoadError(
            f"{context}: counts.skeleton_points={expected_skeleton_count} does not match "
            f"the loaded skeleton point count {len(skeleton_points_xy)}."
        )

    expected_stenosis_count = counts.get("stenosis_points")
    if expected_stenosis_count is not None and int(expected_stenosis_count) != len(observations):
        raise TemporalLoadError(
            f"{context}: counts.stenosis_points={expected_stenosis_count} does not match "
            f"the loaded stenosis point count {len(observations)}."
        )


def _group_frame_results_by_view(frame_results: list[FrameLevelResult]) -> dict[str, list[FrameLevelResult]]:
    grouped_frames: dict[str, list[FrameLevelResult]] = defaultdict(list)
    for frame_result in frame_results:
        if not frame_result.view_id:
            raise TemporalLoadError(
                f"{frame_result.result_path}: missing view_id; cannot group frame results into view sequences."
            )
        grouped_frames[frame_result.view_id].append(frame_result)

    return dict(grouped_frames)


def _validate_view_sequence(
    view_id: str,
    frame_results: list[FrameLevelResult],
    *,
    expected_frame_count: int | None,
) -> None:
    if expected_frame_count is not None and len(frame_results) != expected_frame_count:
        raise TemporalLoadError(
            f"View '{view_id}' has {len(frame_results)} frame results; expected {expected_frame_count}."
        )

    seen_frame_indices: set[int] = set()
    reference_size: tuple[int, int] | None = None

    for frame_result in frame_results:
        if frame_result.frame_index is None:
            raise TemporalLoadError(
                f"{frame_result.result_path}: frame_index is required to preserve temporal ordering."
            )
        if frame_result.frame_index in seen_frame_indices:
            raise TemporalLoadError(
                f"View '{view_id}' contains duplicate frame_index values. Duplicate: {frame_result.frame_index}."
            )
        seen_frame_indices.add(frame_result.frame_index)

        current_size = (int(frame_result.width), int(frame_result.height))
        if reference_size is None:
            reference_size = current_size
            continue
        if current_size != reference_size:
            raise TemporalLoadError(
                f"View '{view_id}' contains inconsistent image sizes. "
                f"Expected {reference_size[0]}x{reference_size[1]}, got {current_size[0]}x{current_size[1]} "
                f"for {frame_result.image_name}."
            )
