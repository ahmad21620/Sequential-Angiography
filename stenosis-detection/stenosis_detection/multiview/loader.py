from __future__ import annotations

from dataclasses import replace
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any

from .models import (
    LoadedMultiViewCase,
    LoadedMultiViewView,
    MultiViewCaseInput,
    MultiViewViewInput,
    ViewLevelLesionCandidate,
)


class MultiViewLoadError(ValueError):
    pass


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TEMPORAL_RESULT_FILENAME = "view_temporal_fusion.json"


def load_multiview_case_input(
    case_input_path: str | Path,
    *,
    temporal_results_root: str | Path | None = None,
    case_root_tree: str | Path | None = None,
) -> MultiViewCaseInput:
    """Load one case-level multiview input JSON file."""
    resolved_path = Path(case_input_path)
    payload = _read_json_object(resolved_path)
    context = str(resolved_path)
    case_id = _require_non_empty_string(payload, "case_id", context=context)
    raw_views = _require_array(payload, "views", context=context)
    if not raw_views:
        raise MultiViewLoadError(f"{context}: field 'views' must contain at least one view.")

    views = [
        _load_view_input(
            item,
            base_dir=resolved_path.parent,
            context=f"{context} -> views[{index}]",
        )
        for index, item in enumerate(raw_views)
    ]
    if temporal_results_root is not None:
        views = _override_temporal_fusion_paths(
            views,
            case_id=case_id,
            case_input_path=resolved_path,
            temporal_results_root=Path(temporal_results_root),
            case_root_tree=None if case_root_tree is None else Path(case_root_tree),
        )
    _validate_unique_view_ids(views, context=context)
    return MultiViewCaseInput(case_id=case_id, views=views)


def load_multiview_view_result(view_input: MultiViewViewInput) -> LoadedMultiViewView:
    """Load one temporal fusion JSON referenced by a view entry."""
    payload = _read_json_object(view_input.temporal_fusion_json_path)
    context = str(view_input.temporal_fusion_json_path)
    temporal_view_id = _require_non_empty_string(payload, "view_id", context=context)
    if not _temporal_view_matches_input(temporal_view_id, view_input):
        raise MultiViewLoadError(
            f"{context}: temporal fusion view_id '{temporal_view_id}' does not match "
            f"multiview input view_id '{view_input.view_id}' or sequence_id '{view_input.sequence_id}'."
        )

    persistent_payloads = _require_array(payload, "persistent_lesions", context=context)
    persistent_lesions = [
        _load_lesion_candidate(item, context=f"{context} -> persistent_lesions[{index}]")
        for index, item in enumerate(persistent_payloads)
    ]

    raw_final_lesion = payload.get("final_lesion")
    final_lesion = None
    if raw_final_lesion is not None:
        final_lesion = _load_lesion_candidate(raw_final_lesion, context=f"{context} -> final_lesion")

    return LoadedMultiViewView(
        view_input=view_input,
        final_lesion=final_lesion,
        persistent_lesions=persistent_lesions,
    )


def load_multiview_case(
    case_input_path: str | Path,
    *,
    temporal_results_root: str | Path | None = None,
    case_root_tree: str | Path | None = None,
) -> LoadedMultiViewCase:
    """Load one case input plus all referenced temporal fusion outputs."""
    case_input = load_multiview_case_input(
        case_input_path,
        temporal_results_root=temporal_results_root,
        case_root_tree=case_root_tree,
    )
    available_views = _filter_available_views(case_input.views, case_input_path=Path(case_input_path))
    return LoadedMultiViewCase(
        case_id=case_input.case_id,
        views=[load_multiview_view_result(view_input) for view_input in available_views],
    )


def _load_view_input(payload: object, *, base_dir: Path, context: str) -> MultiViewViewInput:
    if not isinstance(payload, dict):
        raise MultiViewLoadError(f"{context}: expected a JSON object, got {type(payload)!r}")

    raw_temporal_fusion_json = _require_non_empty_string(payload, "temporal_fusion_json", context=context)

    return MultiViewViewInput(
        view_id=_require_non_empty_string(payload, "view_id", context=context),
        sequence_id=_require_non_empty_string(payload, "sequence_id", context=context),
        rao_lao=_require_finite_float(payload, "rao_lao", context=context),
        cra_cau=_require_finite_float(payload, "cra_cau", context=context),
        temporal_fusion_json_path=_resolve_temporal_fusion_json_path(base_dir, raw_temporal_fusion_json),
    )


def _load_lesion_candidate(payload: object, *, context: str) -> ViewLevelLesionCandidate:
    if not isinstance(payload, dict):
        raise MultiViewLoadError(f"{context}: expected a JSON object, got {type(payload)!r}")

    degrees_payload = _require_object(payload, "degrees", context=context)
    positions_payload = _require_object(payload, "positions", context=context)
    stability_payload = _require_object(payload, "stability", context=context)
    supporting_frame_count = _require_positive_int(payload, "supporting_frame_count", context=context)
    total_frame_count = _require_positive_int(payload, "total_frame_count", context=context)
    frame_indices = _load_frame_indices(payload, context=context)
    median_degree = _require_finite_float(degrees_payload, "median", context=f"{context} -> degrees")
    max_degree = _require_finite_float(degrees_payload, "max", context=f"{context} -> degrees")
    degree_std = _require_non_negative_float(stability_payload, "degree_std", context=f"{context} -> stability")
    max_frame_gap = _require_non_negative_int(stability_payload, "max_frame_gap", context=f"{context} -> stability")

    if len(frame_indices) != supporting_frame_count:
        raise MultiViewLoadError(
            f"{context}: supporting_frame_count={supporting_frame_count} does not match "
            f"the number of frame_indices ({len(frame_indices)})."
        )
    if supporting_frame_count > total_frame_count:
        raise MultiViewLoadError(
            f"{context}: supporting_frame_count={supporting_frame_count} cannot exceed "
            f"total_frame_count={total_frame_count}."
        )
    if max_degree < median_degree:
        raise MultiViewLoadError(
            f"{context}: degrees.max={max_degree} cannot be smaller than degrees.median={median_degree}."
        )

    return ViewLevelLesionCandidate(
        lesion_id=_require_positive_int(payload, "lesion_id", context=context),
        track_id=_require_positive_int(payload, "track_id", context=context),
        severity=_require_non_empty_string(payload, "severity", context=context),
        supporting_frame_count=supporting_frame_count,
        total_frame_count=total_frame_count,
        persistence_ratio=_require_probability(payload, "persistence_ratio", context=context),
        frame_indices=frame_indices,
        median_degree=median_degree,
        max_degree=max_degree,
        degree_std=degree_std,
        max_frame_gap=max_frame_gap,
        median_centerline_position=_load_optional_finite_float(
            positions_payload,
            "median_centerline_position",
            context=f"{context} -> positions",
        ),
        dominant_centerline_component_id=_load_optional_positive_int(
            positions_payload,
            "dominant_centerline_component_id",
            context=f"{context} -> positions",
        ),
    )


def _load_frame_indices(payload: dict[str, Any], *, context: str) -> list[int]:
    frame_indices = _require_array(payload, "frame_indices", context=context)
    return [_coerce_positive_int(frame_index, field_name="frame_indices", context=context) for frame_index in frame_indices]


def _validate_unique_view_ids(views: list[MultiViewViewInput], *, context: str) -> None:
    seen_view_ids: set[str] = set()
    for view in views:
        if view.view_id in seen_view_ids:
            raise MultiViewLoadError(f"{context}: duplicate view_id '{view.view_id}' is not allowed.")
        seen_view_ids.add(view.view_id)


def _filter_available_views(
    views: list[MultiViewViewInput],
    *,
    case_input_path: Path,
) -> list[MultiViewViewInput]:
    case_root = case_input_path.parent
    case_view_keys = _discover_case_view_directory_keys(case_root)
    enforce_case_view_match = (
        case_input_path.name.lower() == "views.json"
        or any(_view_matches_case_directory(view, case_view_keys) for view in views)
    )

    available_views: list[MultiViewViewInput] = []
    for view in views:
        if enforce_case_view_match and not _view_matches_case_directory(view, case_view_keys):
            _warn_skipped_view(
                view,
                f"matching view folder not found under case root: {case_root}",
            )
            continue
        if not view.temporal_fusion_json_path.is_file():
            _warn_skipped_view(
                view,
                f"temporal fusion JSON not found: {view.temporal_fusion_json_path}",
            )
            continue
        available_views.append(view)

    if not available_views:
        raise MultiViewLoadError(
            f"{case_input_path}: no usable views were found. "
            "Check that listed views still exist in the case and have temporal fusion JSON outputs."
        )

    return available_views


def _warn_skipped_view(view: MultiViewViewInput, reason: str) -> None:
    print(f"Warning: skipping view '{view.view_id}': {reason}", file=sys.stderr)


def _discover_case_view_directory_keys(case_root: Path) -> set[str]:
    if not case_root.is_dir():
        return set()

    keys: set[str] = set()
    for directory in _iter_case_image_directories(case_root):
        relative_directory = _normalize_view_identifier(directory.relative_to(case_root).as_posix())
        keys.add(relative_directory)
        keys.add(_normalize_view_identifier(directory.name))

        if directory.name.lower() == "frames" and directory.parent != case_root:
            relative_parent = _normalize_view_identifier(directory.parent.relative_to(case_root).as_posix())
            keys.add(relative_parent)
            keys.add(_normalize_view_identifier(directory.parent.name))

    return {key for key in keys if key}


def _iter_case_image_directories(case_root: Path):
    for directory in case_root.rglob("*"):
        if not directory.is_dir():
            continue
        if _directory_contains_supported_images(directory):
            yield directory


def _directory_contains_supported_images(directory: Path) -> bool:
    try:
        return any(
            child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            for child in directory.iterdir()
        )
    except OSError:
        return False


def _view_matches_case_directory(view: MultiViewViewInput, case_view_keys: set[str]) -> bool:
    if not case_view_keys:
        return False
    return any(identifier in case_view_keys for identifier in _view_directory_identifiers(view))


def _view_directory_identifiers(view: MultiViewViewInput) -> set[str]:
    identifiers: set[str] = set()
    for raw_identifier in (view.view_id, view.sequence_id):
        normalized_identifier = _normalize_view_identifier(raw_identifier)
        if not normalized_identifier:
            continue
        identifiers.add(normalized_identifier)
        identifiers.add(normalized_identifier.rsplit("/", 1)[-1])
    return identifiers


def _normalize_view_identifier(identifier: str) -> str:
    return identifier.replace("\\", "/").strip("/")


def _resolve_temporal_fusion_json_path(base_dir: Path, raw_path: str) -> Path:
    candidate_path = Path(raw_path)
    if not candidate_path.is_absolute():
        candidate_path = base_dir / candidate_path
    return candidate_path.resolve()


def _override_temporal_fusion_paths(
    views: list[MultiViewViewInput],
    *,
    case_id: str,
    case_input_path: Path,
    temporal_results_root: Path,
    case_root_tree: Path | None,
) -> list[MultiViewViewInput]:
    resolved_temporal_root = temporal_results_root.resolve()
    case_relative_paths = _temporal_case_relative_path_candidates(
        case_id=case_id,
        case_input_path=case_input_path,
        case_root_tree=case_root_tree,
    )

    overridden_views: list[MultiViewViewInput] = []
    for view in views:
        candidates = _temporal_override_candidates(
            view,
            temporal_results_root=resolved_temporal_root,
            case_relative_paths=case_relative_paths,
        )
        selected_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        overridden_views.append(replace(view, temporal_fusion_json_path=selected_path.resolve()))

    return overridden_views


def _temporal_case_relative_path_candidates(
    *,
    case_id: str,
    case_input_path: Path,
    case_root_tree: Path | None,
) -> list[Path]:
    candidates: list[Path] = []
    if case_root_tree is not None:
        try:
            candidates.append(case_input_path.parent.resolve().relative_to(case_root_tree.resolve()))
        except ValueError as exc:
            raise MultiViewLoadError(
                f"{case_input_path}: case input is not under --case-root-tree '{case_root_tree}'."
            ) from exc

    candidates.extend(_identifier_to_relative_paths(case_id))
    candidates.append(Path(case_input_path.parent.name))
    return _deduplicate_relative_paths(candidates)


def _temporal_override_candidates(
    view: MultiViewViewInput,
    *,
    temporal_results_root: Path,
    case_relative_paths: list[Path],
) -> list[Path]:
    view_relative_paths = _deduplicate_relative_paths(
        [
            *_identifier_to_relative_paths(view.sequence_id),
            *_identifier_to_relative_paths(view.view_id),
        ]
    )
    candidates: list[Path] = []
    for case_relative_path in case_relative_paths:
        for view_relative_path in view_relative_paths:
            candidates.append(temporal_results_root / case_relative_path / view_relative_path / TEMPORAL_RESULT_FILENAME)
    for view_relative_path in view_relative_paths:
        candidates.append(temporal_results_root / view_relative_path / TEMPORAL_RESULT_FILENAME)
    return _deduplicate_absolute_paths(candidates)


def _identifier_to_relative_paths(identifier: str) -> list[Path]:
    normalized_identifier = _normalize_view_identifier(identifier)
    if not normalized_identifier:
        return []

    parts = _safe_path_parts(normalized_identifier)
    if not parts:
        return []

    paths = [Path(*parts)]
    if len(parts) > 1:
        paths.append(Path(parts[-1]))
    return paths


def _safe_path_parts(identifier: str) -> list[str]:
    return [part for part in identifier.replace("\\", "/").split("/") if part not in {"", ".", ".."}]


def _deduplicate_relative_paths(paths: list[Path]) -> list[Path]:
    deduplicated: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        clean_parts = [part for part in path.parts if part not in {"", ".", ".."}]
        if not clean_parts:
            continue
        clean_path = Path(*clean_parts)
        key = clean_path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(clean_path)
    return deduplicated


def _deduplicate_absolute_paths(paths: list[Path]) -> list[Path]:
    deduplicated: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(path)
    return deduplicated


def _temporal_view_matches_input(temporal_view_id: str, view_input: MultiViewViewInput) -> bool:
    if temporal_view_id == view_input.view_id:
        return True

    normalized_temporal_view_id = temporal_view_id.replace("\\", "/").strip("/")
    if not normalized_temporal_view_id:
        return False

    temporal_sequence_id = normalized_temporal_view_id.rsplit("/", 1)[-1]
    return temporal_sequence_id == view_input.sequence_id


def _read_json_object(json_path: Path) -> dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {json_path}")
    if not json_path.is_file():
        raise MultiViewLoadError(f"JSON path is not a file: {json_path}")

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MultiViewLoadError(f"{json_path}: invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise MultiViewLoadError(f"{json_path}: expected top-level JSON object, got {type(payload)!r}")

    return payload


def _require_object(payload: dict[str, Any], field_name: str, *, context: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise MultiViewLoadError(f"{context}: missing required object field '{field_name}'.")
    return value


def _require_array(payload: dict[str, Any], field_name: str, *, context: str) -> list[Any]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise MultiViewLoadError(f"{context}: missing required array field '{field_name}'.")
    return value


def _require_non_empty_string(payload: dict[str, Any], field_name: str, *, context: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise MultiViewLoadError(f"{context}: missing required string field '{field_name}'.")
    return value


def _require_positive_int(payload: dict[str, Any], field_name: str, *, context: str) -> int:
    return _coerce_positive_int(payload.get(field_name), field_name=field_name, context=context)


def _require_non_negative_int(payload: dict[str, Any], field_name: str, *, context: str) -> int:
    return _coerce_non_negative_int(payload.get(field_name), field_name=field_name, context=context)


def _coerce_positive_int(value: object, *, field_name: str, context: str) -> int:
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise MultiViewLoadError(f"{context}: field '{field_name}' must be an integer.") from exc

    if int_value < 1:
        raise MultiViewLoadError(f"{context}: field '{field_name}' must be >= 1.")

    return int_value


def _coerce_non_negative_int(value: object, *, field_name: str, context: str) -> int:
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise MultiViewLoadError(f"{context}: field '{field_name}' must be an integer.") from exc

    if int_value < 0:
        raise MultiViewLoadError(f"{context}: field '{field_name}' must be >= 0.")

    return int_value


def _require_finite_float(payload: dict[str, Any], field_name: str, *, context: str) -> float:
    float_value = _coerce_float(payload.get(field_name), field_name=field_name, context=context)
    if not isfinite(float_value):
        raise MultiViewLoadError(f"{context}: field '{field_name}' must be finite.")
    return float_value


def _require_non_negative_float(payload: dict[str, Any], field_name: str, *, context: str) -> float:
    float_value = _require_finite_float(payload, field_name, context=context)
    if float_value < 0.0:
        raise MultiViewLoadError(f"{context}: field '{field_name}' must be >= 0.0.")
    return float_value


def _require_probability(payload: dict[str, Any], field_name: str, *, context: str) -> float:
    probability = _require_finite_float(payload, field_name, context=context)
    if not 0.0 <= probability <= 1.0:
        raise MultiViewLoadError(f"{context}: field '{field_name}' must be in the range [0.0, 1.0].")
    return probability


def _load_optional_finite_float(payload: dict[str, Any], field_name: str, *, context: str) -> float | None:
    value = payload.get(field_name)
    if value is None:
        return None

    float_value = _coerce_float(value, field_name=field_name, context=context)
    if not isfinite(float_value):
        raise MultiViewLoadError(f"{context}: field '{field_name}' must be finite.")
    return float_value


def _load_optional_positive_int(payload: dict[str, Any], field_name: str, *, context: str) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    return _coerce_positive_int(value, field_name=field_name, context=context)


def _coerce_float(value: object, *, field_name: str, context: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MultiViewLoadError(f"{context}: field '{field_name}' must be a float.") from exc
