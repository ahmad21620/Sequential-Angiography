from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path

from .models import (
    ConfidenceSummary,
    FinalCaseLesion,
    LesionCandidateScore,
    LoadedMultiViewCase,
    LoadedMultiViewView,
    MultiViewCaseResult,
    MultiViewFusionConfig,
    MultiViewFusionMetadata,
    MultiViewPerViewSummary,
    MultiViewViewInput,
    ViewLevelLesionCandidate,
)


DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES = 20.0
DISTINCT_VIEW_ANGLE_DISTANCE_DEGREES = 45.0
MIN_SUPPORTING_VIEW_SCORE = 0.15
SUPPORT_SCORE_SCALE = 0.20
MAX_DISTINCT_VIEW_SUPPORT_SCORE = 0.35
MEDIUM_CONFIDENCE_THRESHOLD = 0.45
HIGH_CONFIDENCE_THRESHOLD = 0.75
VIEW_DIVERSITY_MODES = ("angle", "projection_group", "auto")
LEFT_PROJECTION_GROUPS = frozenset({"LCA", "LCA2"})
FINAL_CASE_SELECTION_RULE = (
    "highest_total_score_then_more_distinct_support_then_higher_candidate_score_then_higher_median_degree_"
    "then_higher_persistence_ratio_then_view_id_then_lesion_id"
)


@dataclass(slots=True)
class _CandidateReference:
    view_result: LoadedMultiViewView
    candidate: ViewLevelLesionCandidate
    score: LesionCandidateScore


def run_multiview_fusion(
    multiview_case: LoadedMultiViewCase,
    *,
    config: MultiViewFusionConfig | None = None,
) -> MultiViewCaseResult:
    """Run deterministic late fusion across one case of view-level results."""
    resolved_config = _resolve_fusion_config(config)
    view_summaries = build_multiview_view_summaries(multiview_case)
    final_case_lesion = select_final_case_lesion(
        multiview_case,
        view_summaries=view_summaries,
        config=resolved_config,
    )

    if final_case_lesion is None:
        confidence = ConfidenceSummary(score=0.0, label="low")
        supporting_views: list[str] = []
    else:
        confidence = final_case_lesion.confidence
        supporting_views = list(final_case_lesion.supporting_view_ids)

    return MultiViewCaseResult(
        case_id=multiview_case.case_id,
        views=[view_result.view_input for view_result in multiview_case.views],
        per_view_summary=view_summaries,
        final_case_lesion=final_case_lesion,
        confidence=confidence,
        supporting_views=supporting_views,
        fusion_metadata=MultiViewFusionMetadata(
            selection_rule=FINAL_CASE_SELECTION_RULE,
            total_candidate_count=sum(summary.candidate_count for summary in view_summaries),
            config=resolved_config,
        ),
    )


def save_multiview_case_result(case_result: MultiViewCaseResult, output_path: str | Path) -> Path:
    """Write one case-level multiview fusion JSON file."""
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(case_result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return resolved_output_path


def build_multiview_view_summaries(multiview_case: LoadedMultiViewCase) -> list[MultiViewPerViewSummary]:
    """Summarize the strongest candidate found in each view."""
    summaries: list[MultiViewPerViewSummary] = []
    for view_result in multiview_case.views:
        candidate_references = _collect_view_candidate_references(view_result)
        best_candidate_reference = None if not candidate_references else min(candidate_references, key=_candidate_reference_sort_key)
        summaries.append(
            MultiViewPerViewSummary(
                view_input=view_result.view_input,
                candidate_source=view_result.candidate_source,
                candidate_count=len(candidate_references),
                best_candidate=None if best_candidate_reference is None else best_candidate_reference.candidate,
                best_candidate_score=None if best_candidate_reference is None else best_candidate_reference.score,
            )
        )

    return summaries


def compute_angle_distance(first_view: MultiViewViewInput, second_view: MultiViewViewInput) -> float:
    rao_lao_delta = _circular_angle_delta_degrees(first_view.rao_lao, second_view.rao_lao)
    cra_cau_delta = _circular_angle_delta_degrees(first_view.cra_cau, second_view.cra_cau)
    return float(sqrt((rao_lao_delta * rao_lao_delta) + (cra_cau_delta * cra_cau_delta)))


def is_duplicate_view(
    first_view: MultiViewViewInput,
    second_view: MultiViewViewInput,
    *,
    duplicate_distance_degrees: float = DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES,
) -> bool:
    return compute_angle_distance(first_view, second_view) <= duplicate_distance_degrees


def compute_distinct_view_support_score(
    primary_view: MultiViewViewInput,
    other_view_summaries: list[MultiViewPerViewSummary],
    *,
    duplicate_distance_degrees: float = DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES,
    distinct_distance_degrees: float = DISTINCT_VIEW_ANGLE_DISTANCE_DEGREES,
    support_score_scale: float = SUPPORT_SCORE_SCALE,
    min_supporting_view_score: float = MIN_SUPPORTING_VIEW_SCORE,
    max_support_score: float = MAX_DISTINCT_VIEW_SUPPORT_SCORE,
    view_diversity_mode: str = "angle",
) -> tuple[float, list[str], list[str]]:
    _validate_view_diversity_mode(view_diversity_mode)
    supporting_view_ids: list[str] = []
    distinct_supporting_view_ids: list[str] = []
    support_score = 0.0

    for view_summary in other_view_summaries:
        if view_summary.view_input.view_id == primary_view.view_id:
            continue
        if view_summary.best_candidate is None or view_summary.best_candidate_score is None:
            continue

        candidate_score = view_summary.best_candidate_score.candidate_score
        if candidate_score < min_supporting_view_score:
            continue

        diversity_weight = compute_view_diversity_weight(
            primary_view,
            view_summary.view_input,
            view_diversity_mode,
            duplicate_distance_degrees=duplicate_distance_degrees,
            distinct_distance_degrees=distinct_distance_degrees,
        )
        if diversity_weight <= 0.0:
            continue

        support_score += support_score_scale * candidate_score * diversity_weight
        supporting_view_ids.append(view_summary.view_input.view_id)
        if is_distinct_supporting_view(
            primary_view,
            view_summary.view_input,
            view_diversity_mode,
            duplicate_distance_degrees=duplicate_distance_degrees,
        ):
            distinct_supporting_view_ids.append(view_summary.view_input.view_id)

    return (
        min(max_support_score, support_score),
        supporting_view_ids,
        distinct_supporting_view_ids,
    )


def compute_diversity_weight(
    angle_distance: float,
    *,
    duplicate_distance_degrees: float = DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES,
    distinct_distance_degrees: float = DISTINCT_VIEW_ANGLE_DISTANCE_DEGREES,
) -> float:
    """Return the angle-diversity multiplier used for cross-view support."""
    if angle_distance <= duplicate_distance_degrees:
        return 0.25
    if angle_distance >= distinct_distance_degrees:
        return 1.0

    distance_range = distinct_distance_degrees - duplicate_distance_degrees
    normalized_distance = (angle_distance - duplicate_distance_degrees) / distance_range
    return 0.25 + (0.75 * normalized_distance)


def compute_view_diversity_weight(
    first_view: MultiViewViewInput,
    second_view: MultiViewViewInput,
    view_diversity_mode: str = "angle",
    *,
    duplicate_distance_degrees: float = DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES,
    distinct_distance_degrees: float = DISTINCT_VIEW_ANGLE_DISTANCE_DEGREES,
) -> float:
    pair_mode = _resolve_pair_view_diversity_mode(first_view, second_view, view_diversity_mode)
    if pair_mode == "projection_group":
        return compute_projection_group_diversity_weight(first_view, second_view)

    angle_distance = compute_angle_distance(first_view, second_view)
    return compute_diversity_weight(
        angle_distance,
        duplicate_distance_degrees=duplicate_distance_degrees,
        distinct_distance_degrees=distinct_distance_degrees,
    )


def is_distinct_supporting_view(
    first_view: MultiViewViewInput,
    second_view: MultiViewViewInput,
    view_diversity_mode: str = "angle",
    *,
    duplicate_distance_degrees: float = DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES,
) -> bool:
    pair_mode = _resolve_pair_view_diversity_mode(first_view, second_view, view_diversity_mode)
    if pair_mode == "projection_group":
        return is_projection_group_distinct_support(first_view, second_view)
    return not is_duplicate_view(
        first_view,
        second_view,
        duplicate_distance_degrees=duplicate_distance_degrees,
    )


def compute_projection_group_diversity_weight(
    first_view: MultiViewViewInput,
    second_view: MultiViewViewInput,
) -> float:
    first_side = _projection_side(first_view)
    second_side = _projection_side(second_view)
    if first_side == "unknown" or second_side == "unknown" or first_side != second_side:
        return 0.0

    first_groups = _projection_group_set(first_view)
    second_groups = _projection_group_set(second_view)
    if not first_groups or not second_groups:
        return 0.0

    if first_groups == second_groups and len(first_groups) == 1:
        return 0.25

    if first_side == "left" and first_groups <= LEFT_PROJECTION_GROUPS and second_groups <= LEFT_PROJECTION_GROUPS:
        return 0.65

    return 0.0


def is_projection_group_distinct_support(
    first_view: MultiViewViewInput,
    second_view: MultiViewViewInput,
) -> bool:
    if _projection_side(first_view) != "left" or _projection_side(second_view) != "left":
        return False
    first_groups = _projection_group_set(first_view)
    second_groups = _projection_group_set(second_view)
    if len(first_groups) != 1 or len(second_groups) != 1:
        return False
    if first_view.projection_status != "known" or second_view.projection_status != "known":
        return False
    return first_groups != second_groups and first_groups <= LEFT_PROJECTION_GROUPS and second_groups <= LEFT_PROJECTION_GROUPS


def score_lesion_candidate(candidate: ViewLevelLesionCandidate) -> LesionCandidateScore:
    base_score = candidate.base_score
    stability_adjustment = _compute_stability_adjustment(candidate)
    candidate_score = _clamp_score(base_score + stability_adjustment)
    return LesionCandidateScore(
        base_score=base_score,
        stability_adjustment=stability_adjustment,
        candidate_score=candidate_score,
    )


def select_final_case_lesion(
    multiview_case: LoadedMultiViewCase,
    *,
    view_summaries: list[MultiViewPerViewSummary] | None = None,
    config: MultiViewFusionConfig | None = None,
) -> FinalCaseLesion | None:
    """Select the final case-level lesion from all view candidates."""
    resolved_config = _resolve_fusion_config(config)
    if view_summaries is None:
        view_summaries = build_multiview_view_summaries(multiview_case)

    candidate_references = _collect_case_candidate_references(multiview_case)
    if not candidate_references:
        return None

    final_case_lesions = [
        _build_final_case_lesion(
            candidate_reference,
            view_summaries=view_summaries,
            config=resolved_config,
        )
        for candidate_reference in candidate_references
    ]
    return min(final_case_lesions, key=_final_case_lesion_sort_key)


def _collect_case_candidate_references(multiview_case: LoadedMultiViewCase) -> list[_CandidateReference]:
    candidate_references: list[_CandidateReference] = []
    for view_result in multiview_case.views:
        candidate_references.extend(_collect_view_candidate_references(view_result))
    return candidate_references


def _collect_view_candidate_references(view_result: LoadedMultiViewView) -> list[_CandidateReference]:
    return [
        _CandidateReference(
            view_result=view_result,
            candidate=candidate,
            score=score_lesion_candidate(candidate),
        )
        for candidate in view_result.candidates
    ]


def _build_final_case_lesion(
    candidate_reference: _CandidateReference,
    *,
    view_summaries: list[MultiViewPerViewSummary],
    config: MultiViewFusionConfig,
) -> FinalCaseLesion:
    distinct_view_support_score, supporting_view_ids, distinct_supporting_view_ids = compute_distinct_view_support_score(
        candidate_reference.view_result.view_input,
        view_summaries,
        duplicate_distance_degrees=config.duplicate_view_angle_distance_degrees,
        distinct_distance_degrees=config.distinct_view_angle_distance_degrees,
        support_score_scale=config.support_score_scale,
        view_diversity_mode=config.view_diversity_mode,
    )
    total_score = candidate_reference.score.candidate_score + distinct_view_support_score
    confidence_score = _clamp_score(total_score)

    return FinalCaseLesion(
        primary_view_id=candidate_reference.view_result.view_input.view_id,
        primary_sequence_id=candidate_reference.view_result.view_input.sequence_id,
        lesion=candidate_reference.candidate,
        score_breakdown=candidate_reference.score,
        distinct_view_support_score=distinct_view_support_score,
        total_score=total_score,
        supporting_view_ids=supporting_view_ids,
        distinct_supporting_view_ids=distinct_supporting_view_ids,
        confidence=ConfidenceSummary(
            score=confidence_score,
            label=_label_confidence(confidence_score, config),
        ),
    )


def _compute_stability_adjustment(candidate: ViewLevelLesionCandidate) -> float:
    degree_component = 0.05 - min(0.10, 0.40 * candidate.degree_std)
    gap_component = 0.03 - min(0.08, 0.04 * max(candidate.max_frame_gap - 1, 0))
    return _clamp_signed_score(degree_component + gap_component, lower_bound=-0.10, upper_bound=0.10)


def _label_confidence(confidence_score: float, config: MultiViewFusionConfig) -> str:
    if confidence_score >= config.high_confidence_threshold:
        return "high"
    if confidence_score >= config.medium_confidence_threshold:
        return "medium"
    return "low"


def _resolve_fusion_config(config: MultiViewFusionConfig | None) -> MultiViewFusionConfig:
    resolved_config = MultiViewFusionConfig() if config is None else config
    _validate_fusion_config(resolved_config)
    return resolved_config


def _validate_fusion_config(config: MultiViewFusionConfig) -> None:
    _validate_view_diversity_mode(config.view_diversity_mode)
    if config.duplicate_view_angle_distance_degrees <= 0.0:
        raise ValueError("duplicate_view_angle_distance_degrees must be positive.")
    if config.distinct_view_angle_distance_degrees <= config.duplicate_view_angle_distance_degrees:
        raise ValueError("distinct_view_angle_distance_degrees must be greater than duplicate_view_angle_distance_degrees.")
    if config.support_score_scale < 0.0:
        raise ValueError("support_score_scale must be >= 0.0.")
    if not 0.0 <= config.medium_confidence_threshold <= 1.0:
        raise ValueError("medium_confidence_threshold must be in the range [0.0, 1.0].")
    if not 0.0 <= config.high_confidence_threshold <= 1.0:
        raise ValueError("high_confidence_threshold must be in the range [0.0, 1.0].")
    if config.medium_confidence_threshold > config.high_confidence_threshold:
        raise ValueError("medium_confidence_threshold must be <= high_confidence_threshold.")


def _validate_view_diversity_mode(view_diversity_mode: str) -> None:
    if view_diversity_mode not in VIEW_DIVERSITY_MODES:
        raise ValueError(f"view_diversity_mode must be one of {VIEW_DIVERSITY_MODES}.")


def _resolve_pair_view_diversity_mode(
    first_view: MultiViewViewInput,
    second_view: MultiViewViewInput,
    view_diversity_mode: str,
) -> str:
    if view_diversity_mode != "auto":
        return view_diversity_mode
    if _has_projection_metadata(first_view) or _has_projection_metadata(second_view):
        return "projection_group"
    return "angle"


def _has_projection_metadata(view: MultiViewViewInput) -> bool:
    return (
        view.angle_status is not None
        or view.projection_group is not None
        or view.projection_groups is not None
        or view.coronary_side is not None
        or view.projection_status is not None
    )


def _projection_side(view: MultiViewViewInput) -> str:
    side = (view.coronary_side or "unknown").strip().lower()
    if side in {"left", "right"}:
        return side
    return "unknown"


def _projection_group_set(view: MultiViewViewInput) -> frozenset[str]:
    groups: list[str] = []
    if view.projection_groups:
        groups.extend(view.projection_groups)
    if view.projection_group and view.projection_group != "unknown":
        groups.append(view.projection_group)
    normalized_groups = {group.strip().upper() for group in groups if group.strip()}
    return frozenset(group for group in normalized_groups if group in {"LCA", "LCA2", "RCA"})


def _clamp_score(score: float) -> float:
    return _clamp_signed_score(score, lower_bound=0.0, upper_bound=1.0)


def _clamp_signed_score(score: float, *, lower_bound: float, upper_bound: float) -> float:
    return float(max(lower_bound, min(upper_bound, score)))


def _circular_angle_delta_degrees(first_angle: float, second_angle: float) -> float:
    raw_delta = abs(float(first_angle) - float(second_angle)) % 360.0
    return min(raw_delta, 360.0 - raw_delta)


def _candidate_reference_sort_key(candidate_reference: _CandidateReference) -> tuple[float, float, float, int]:
    return (
        -candidate_reference.score.candidate_score,
        -candidate_reference.candidate.median_degree,
        -candidate_reference.candidate.persistence_ratio,
        candidate_reference.candidate.lesion_id,
    )


def _final_case_lesion_sort_key(final_case_lesion: FinalCaseLesion) -> tuple[float, int, float, float, float, str, int]:
    return (
        -final_case_lesion.total_score,
        -len(final_case_lesion.distinct_supporting_view_ids),
        -final_case_lesion.score_breakdown.candidate_score,
        -final_case_lesion.lesion.median_degree,
        -final_case_lesion.lesion.persistence_ratio,
        final_case_lesion.primary_view_id,
        final_case_lesion.lesion.lesion_id,
    )


__all__ = [
    "DISTINCT_VIEW_ANGLE_DISTANCE_DEGREES",
    "DUPLICATE_VIEW_ANGLE_DISTANCE_DEGREES",
    "FINAL_CASE_SELECTION_RULE",
    "HIGH_CONFIDENCE_THRESHOLD",
    "MAX_DISTINCT_VIEW_SUPPORT_SCORE",
    "MEDIUM_CONFIDENCE_THRESHOLD",
    "VIEW_DIVERSITY_MODES",
    "SUPPORT_SCORE_SCALE",
    "build_multiview_view_summaries",
    "compute_angle_distance",
    "compute_diversity_weight",
    "compute_distinct_view_support_score",
    "compute_projection_group_diversity_weight",
    "compute_view_diversity_weight",
    "is_duplicate_view",
    "is_distinct_supporting_view",
    "is_projection_group_distinct_support",
    "run_multiview_fusion",
    "save_multiview_case_result",
    "score_lesion_candidate",
    "select_final_case_lesion",
]
