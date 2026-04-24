from __future__ import annotations

from .loader import sort_frame_results
from .models import FrameLevelResult, ReferenceFrameSelection


REFERENCE_SELECTION_METHOD = "largest_centerline_support"
REFERENCE_SCORE_NAME = "skeleton_point_count"


def select_reference_frame(frame_results: list[FrameLevelResult]) -> ReferenceFrameSelection:
    ordered_frames = sort_frame_results(frame_results)
    if not ordered_frames:
        raise ValueError("At least one frame result is required to select a reference frame.")

    middle_position = len(ordered_frames) // 2
    ranked_frames = sorted(
        enumerate(ordered_frames),
        key=lambda item: _reference_frame_ranking_key(
            item[1],
            position=item[0],
            middle_position=middle_position,
        ),
    )

    selected_position, selected_frame = ranked_frames[0]
    support_score = compute_reference_frame_score(selected_frame)
    support_coverage_ratio = _compute_support_coverage_ratio(selected_frame)
    middle_frame_distance = abs(selected_position - middle_position)

    return ReferenceFrameSelection(
        frame=selected_frame,
        method=REFERENCE_SELECTION_METHOD,
        score_name=REFERENCE_SCORE_NAME,
        score_value=float(support_score),
        support_point_count=int(support_score),
        support_coverage_ratio=support_coverage_ratio,
        middle_frame_distance=middle_frame_distance,
        reason=_build_selection_reason(
            selected_frame,
            support_point_count=int(support_score),
            support_coverage_ratio=support_coverage_ratio,
            middle_frame_distance=middle_frame_distance,
        ),
    )


def compute_reference_frame_score(frame_result: FrameLevelResult) -> float:
    return float(len(frame_result.skeleton_points_xy))


def _reference_frame_ranking_key(
    frame_result: FrameLevelResult,
    *,
    position: int,
    middle_position: int,
) -> tuple[float, int, int, str]:
    support_score = compute_reference_frame_score(frame_result)
    middle_frame_distance = abs(position - middle_position)
    frame_index = frame_result.frame_index if frame_result.frame_index is not None else 0

    return (-support_score, middle_frame_distance, frame_index, frame_result.image_name)


def _compute_support_coverage_ratio(frame_result: FrameLevelResult) -> float:
    if frame_result.width is None or frame_result.height is None:
        return 0.0

    image_area = int(frame_result.width) * int(frame_result.height)
    if image_area <= 0:
        return 0.0

    return float(len(frame_result.skeleton_points_xy)) / float(image_area)


def _build_selection_reason(
    frame_result: FrameLevelResult,
    *,
    support_point_count: int,
    support_coverage_ratio: float,
    middle_frame_distance: int,
) -> str:
    return (
        f"Selected {frame_result.image_name} using '{REFERENCE_SELECTION_METHOD}': "
        f"highest skeleton point count ({support_point_count}, coverage {support_coverage_ratio:.6f}). "
        f"Ties are resolved by choosing the frame closest to the sequence middle "
        f"(distance {middle_frame_distance}), then the lower frame index."
    )
